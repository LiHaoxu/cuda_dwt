// quantizer.cu — CUDA midtread quantizer for 3-D CDF 9/7 DWT coefficients
//
// Supports two compression modes:
//   PWE  — point-wise error: q = tolerance * 1.5, with outlier correction
//   PSNR — target PSNR:      q from closed-form MSE estimate, no outliers
//
// Pipeline: subtract mean → pad → DWT → quantize → [inv-quantize → IDWT → outlier detect]
//
// Compilation:
//   cd /home/lihaoxu/wavelet
//   nvcc -std=c++17 -O2 -I include src/quantizer.cu -o quantizer
//
// Usage:
//   ./quantizer <input.raw> <nx> <ny> <nz> <quality> [levels] [prefix]
//               [--mode pwe|psnr]

#include "cdf97_3d_host.cu"
#include "cdf97_3d_f32_host.cu"
#include "quant_kernels.cu"

#include <cub/device/device_scan.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

static constexpr int QUANT_BLOCK = 256;

// ============================================================================
// QuantWorkspace — reusable device buffers for reductions and CUB scan
// ============================================================================

void QuantWorkspace::alloc_reduce(size_t max_blocks)
{
    if (max_blocks <= reduce_buf_elems) return;
    if (d_reduce_buf[0]) cudaFree(d_reduce_buf[0]);
    if (d_reduce_buf[1]) cudaFree(d_reduce_buf[1]);
    reduce_buf_elems = max_blocks;
    CUDA_CHECK(cudaMalloc(&d_reduce_buf[0], max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_reduce_buf[1], max_blocks * sizeof(double)));
}

void QuantWorkspace::ensure_scan_temp(size_t needed)
{
    if (needed <= scan_temp_bytes) return;
    if (d_scan_temp) cudaFree(d_scan_temp);
    scan_temp_bytes = needed;
    CUDA_CHECK(cudaMalloc(&d_scan_temp, needed));
}

void QuantWorkspace::free_all()
{
    if (d_reduce_buf[0]) { cudaFree(d_reduce_buf[0]); d_reduce_buf[0] = nullptr; }
    if (d_reduce_buf[1]) { cudaFree(d_reduce_buf[1]); d_reduce_buf[1] = nullptr; }
    reduce_buf_elems = 0;
    if (d_scan_temp) { cudaFree(d_scan_temp); d_scan_temp = nullptr; }
    scan_temp_bytes = 0;
}

// ============================================================================
// QuantResult::free_device
// ============================================================================

void QuantResult::free_device()
{
    if (d_coeffs)       { cudaFree(d_coeffs);       d_coeffs       = nullptr; }
    if (d_signs)        { cudaFree(d_signs);        d_signs        = nullptr; }
    if (d_outlier_idx)  { cudaFree(d_outlier_idx);  d_outlier_idx  = nullptr; }
    if (d_outlier_val)  { cudaFree(d_outlier_val);  d_outlier_val  = nullptr; }
    if (d_outlier_sign) { cudaFree(d_outlier_sign); d_outlier_sign = nullptr; }
}

// ============================================================================
// Fused pad + subtract-mean kernels (saves one kernel launch for padded data)
// ============================================================================

template <typename Real>
__global__ void kernel_reflect_pad_subtract_mean(
    const Real* __restrict__ src, Real* __restrict__ dst,
    int orig_nx, int orig_ny, int orig_nz,
    int pad_nx,  int pad_ny,  int pad_nz,
    double mean)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t tot = size_t(pad_nx) * pad_ny * pad_nz;
    if (idx >= tot) return;

    int x = static_cast<int>(idx % pad_nx);
    int y = static_cast<int>((idx / pad_nx) % pad_ny);
    int z = static_cast<int>(idx / (size_t(pad_nx) * pad_ny));

    int sx = wavelet::ReflectIdx(x, orig_nx);
    int sy = wavelet::ReflectIdx(y, orig_ny);
    int sz = wavelet::ReflectIdx(z, orig_nz);

    double v = static_cast<double>(src[size_t(sz) * orig_nx * orig_ny + size_t(sy) * orig_nx + sx]);
    dst[idx] = static_cast<Real>(v - mean);
}

template <typename Real>
__global__ void kernel_reflect_pad_subtract_mean_copy(
    const Real* __restrict__ src, Real* __restrict__ dst_pad,
    Real* __restrict__ dst_orig,
    int orig_nx, int orig_ny, int orig_nz,
    int pad_nx,  int pad_ny,  int pad_nz,
    double mean)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t tot = size_t(pad_nx) * pad_ny * pad_nz;
    if (idx >= tot) return;

    int x = static_cast<int>(idx % pad_nx);
    int y = static_cast<int>((idx / pad_nx) % pad_ny);
    int z = static_cast<int>(idx / (size_t(pad_nx) * pad_ny));

    int sx = wavelet::ReflectIdx(x, orig_nx);
    int sy = wavelet::ReflectIdx(y, orig_ny);
    int sz = wavelet::ReflectIdx(z, orig_nz);

    double v = static_cast<double>(src[size_t(sz) * orig_nx * orig_ny + size_t(sy) * orig_nx + sx]);
    Real val = static_cast<Real>(v - mean);
    dst_pad[idx] = val;

    if (x < orig_nx && y < orig_ny && z < orig_nz)
        dst_orig[size_t(z) * orig_nx * orig_ny + size_t(y) * orig_nx + x] = val;
}

// ============================================================================
// GPU reductions
// ============================================================================

struct U8ToU32 {
    __host__ __device__ __forceinline__
    uint32_t operator()(const uint8_t& v) const { return static_cast<uint32_t>(v); }
};

template <typename Real>
static double gpu_compute_mean(const Real* d_data, size_t n, QuantWorkspace& ws)
{
    const int threads = QUANT_BLOCK;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));

    kernel_sum_reduce<Real><<<blocks, threads, threads * sizeof(double)>>>(
        d_data, n, ws.d_reduce_buf[0]);
    CUDA_CHECK(cudaGetLastError());

    int cur = 0;
    while (blocks > 1) {
        int prev = blocks;
        blocks = (prev + threads * 2 - 1) / (threads * 2);
        int next = 1 - cur;
        kernel_sum_reduce<double><<<blocks, threads, threads * sizeof(double)>>>(
            ws.d_reduce_buf[cur], prev, ws.d_reduce_buf[next]);
        CUDA_CHECK(cudaGetLastError());
        cur = next;
    }

    double sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&sum, ws.d_reduce_buf[cur], sizeof(double),
                           cudaMemcpyDeviceToHost));
    return sum / static_cast<double>(n);
}

template <typename Real>
static double gpu_abs_max(const Real* d_data, size_t n, QuantWorkspace& ws)
{
    const int threads = QUANT_BLOCK;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));

    kernel_abs_max<Real><<<blocks, threads, threads * sizeof(double)>>>(
        d_data, n, ws.d_reduce_buf[0]);
    CUDA_CHECK(cudaGetLastError());

    int cur = 0;
    while (blocks > 1) {
        int prev = blocks;
        blocks = (prev + threads * 2 - 1) / (threads * 2);
        int next = 1 - cur;
        kernel_abs_max<double><<<blocks, threads, threads * sizeof(double)>>>(
            ws.d_reduce_buf[cur], prev, ws.d_reduce_buf[next]);
        CUDA_CHECK(cudaGetLastError());
        cur = next;
    }

    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, ws.d_reduce_buf[cur], sizeof(double),
                           cudaMemcpyDeviceToHost));
    return result;
}

// ============================================================================
// Utility functions
// ============================================================================

static int auto_dwt_levels(int nx, int ny, int nz)
{
    int max_dim = std::max({nx, ny, (nz > 1) ? nz : 1});
    if (max_dim >= 512) return 6;
    if (max_dim >= 256) return 5;
    if (max_dim >= 128) return 4;
    if (max_dim >= 64)  return 3;
    if (max_dim >= 32)  return 2;
    return 1;
}

static UIntFlag choose_uint_type(double max_abs, double q)
{
    long long maxll = std::llrint(max_abs / q);
    if (maxll < 0) maxll = -maxll;

    if (maxll <= std::numeric_limits<uint8_t>::max())  return UIntFlag::U8;
    if (maxll <= std::numeric_limits<uint16_t>::max()) return UIntFlag::U16;
    if (maxll <= std::numeric_limits<uint32_t>::max()) return UIntFlag::U32;
    return UIntFlag::U64;
}

static const char* uint_type_name(UIntFlag f)
{
    switch (f) {
        case UIntFlag::U8:  return "uint8";
        case UIntFlag::U16: return "uint16";
        case UIntFlag::U32: return "uint32";
        default:            return "uint64";
    }
}

// PSNR mode: q = 2 * sqrt(3 * MSE), where MSE = range^2 * 10^(-PSNR/10)
static double choose_q_psnr(double data_range, double psnr_db)
{
    double t_mse = data_range * data_range * std::pow(10.0, -psnr_db / 10.0);
    return 2.0 * std::sqrt(3.0 * t_mse);
}

// ============================================================================
// Kernel dispatch helpers (type-erased → typed)
// ============================================================================

template <typename Real>
static void dispatch_quantize(const Real* d_coeffs, void* d_ints,
                              uint64_t* d_signs, size_t n,
                              double inv_q, UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_midtread_quantize<uint8_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint8_t*>(d_ints), d_signs, n, inv_q);
        break;
    case UIntFlag::U16:
        kernel_midtread_quantize<uint16_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint16_t*>(d_ints), d_signs, n, inv_q);
        break;
    case UIntFlag::U32:
        kernel_midtread_quantize<uint32_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint32_t*>(d_ints), d_signs, n, inv_q);
        break;
    default:
        kernel_midtread_quantize<uint64_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint64_t*>(d_ints), d_signs, n, inv_q);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Real>
static void dispatch_inv_quantize(const void* d_ints, const uint64_t* d_signs,
                                  Real* d_coeffs, size_t n,
                                  double q, UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_midtread_inv_quantize<uint8_t, Real><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint8_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U16:
        kernel_midtread_inv_quantize<uint16_t, Real><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint16_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U32:
        kernel_midtread_inv_quantize<uint32_t, Real><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint32_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    default:
        kernel_midtread_inv_quantize<uint64_t, Real><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint64_t*>(d_ints), d_signs, d_coeffs, n, q);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Real>
static void dispatch_outlier_compact(const Real* d_orig, const Real* d_recon,
                                     const uint8_t* d_flags,
                                     const uint32_t* d_offsets,
                                     uint32_t* d_out_idx, void* d_out_val,
                                     uint8_t* d_out_sign,
                                     size_t n, double inv_tol, UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_outlier_compact_u8<uint8_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, d_flags, d_offsets,
            d_out_idx, static_cast<uint8_t*>(d_out_val), d_out_sign, n, inv_tol);
        break;
    case UIntFlag::U16:
        kernel_outlier_compact_u8<uint16_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, d_flags, d_offsets,
            d_out_idx, static_cast<uint16_t*>(d_out_val), d_out_sign, n, inv_tol);
        break;
    case UIntFlag::U32:
        kernel_outlier_compact_u8<uint32_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, d_flags, d_offsets,
            d_out_idx, static_cast<uint32_t*>(d_out_val), d_out_sign, n, inv_tol);
        break;
    default:
        kernel_outlier_compact_u8<uint64_t, Real><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, d_flags, d_offsets,
            d_out_idx, static_cast<uint64_t*>(d_out_val), d_out_sign, n, inv_tol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// File I/O — binary output with headers (26 bytes)
//
// Coefficient / combined file layout (26 bytes header):
//   uint8_t  type_flag    (1)
//   uint64_t num_elements (8)
//   double   q            (8)
//   double   mean         (8)
//   uint8_t  precision    (1)   0=f64, 1=f32
//   [payload: integer array]   (combined: + sign bitmask)
//
// Sign file: raw uint64 bitmask words
//
// Outlier file layout (26 bytes header):
//   uint8_t  type_flag     (1)
//   uint64_t orig_elements (8)
//   double   tolerance     (8)
//   uint64_t num_outliers  (8)
//   uint8_t  precision     (1)   0=f64, 1=f32
//   [sparse payload: indices + abs_values + signs]
// ============================================================================

void save_coefficients(const std::string& path, const QuantResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    uint8_t flag = static_cast<uint8_t>(res.coeff_type);
    std::fwrite(&flag, 1, 1, fp);
    uint64_t ne = res.total_elems;
    std::fwrite(&ne, sizeof(ne), 1, fp);
    std::fwrite(&res.q, sizeof(res.q), 1, fp);
    std::fwrite(&res.mean, sizeof(res.mean), 1, fp);
    uint8_t prec = static_cast<uint8_t>(res.precision);
    std::fwrite(&prec, 1, 1, fp);

    size_t elem_sz = uint_elem_size(res.coeff_type);
    size_t payload = res.total_elems * elem_sz;
    std::vector<uint8_t> buf(payload);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_coeffs, payload,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, payload, fp);
    std::fclose(fp);
}

void save_signs(const std::string& path, const QuantResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    size_t bytes = res.num_words * sizeof(uint64_t);
    std::vector<uint8_t> buf(bytes);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_signs, bytes,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, bytes, fp);
    std::fclose(fp);
}

void save_combined(const std::string& path, const QuantResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    uint8_t flag = static_cast<uint8_t>(res.coeff_type);
    std::fwrite(&flag, 1, 1, fp);
    uint64_t ne = res.total_elems;
    std::fwrite(&ne, sizeof(ne), 1, fp);
    std::fwrite(&res.q, sizeof(res.q), 1, fp);
    std::fwrite(&res.mean, sizeof(res.mean), 1, fp);
    uint8_t prec = static_cast<uint8_t>(res.precision);
    std::fwrite(&prec, 1, 1, fp);

    size_t elem_sz = uint_elem_size(res.coeff_type);
    size_t payload = res.total_elems * elem_sz;
    std::vector<uint8_t> buf(payload);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_coeffs, payload,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, payload, fp);

    size_t sign_bytes = res.num_words * sizeof(uint64_t);
    buf.resize(sign_bytes);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_signs, sign_bytes,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, sign_bytes, fp);

    std::fclose(fp);
}

void save_outliers(const std::string& path, const QuantResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    uint8_t flag = static_cast<uint8_t>(res.outlier_type);
    std::fwrite(&flag, 1, 1, fp);
    uint64_t ne = res.orig_elems;
    std::fwrite(&ne, sizeof(ne), 1, fp);
    double tol = res.q / 1.5;
    std::fwrite(&tol, sizeof(tol), 1, fp);
    std::fwrite(&res.num_outliers, sizeof(res.num_outliers), 1, fp);
    uint8_t prec = static_cast<uint8_t>(res.precision);
    std::fwrite(&prec, 1, 1, fp);

    if (res.num_outliers == 0) {
        std::fclose(fp);
        return;
    }

    const uint64_t n_out = res.num_outliers;
    const size_t elem_sz = uint_elem_size(res.outlier_type);

    {
        std::vector<uint32_t> idx(n_out);
        CUDA_CHECK(cudaMemcpy(idx.data(), res.d_outlier_idx,
                               n_out * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::fwrite(idx.data(), sizeof(uint32_t), n_out, fp);
    }
    {
        std::vector<uint8_t> val(n_out * elem_sz);
        CUDA_CHECK(cudaMemcpy(val.data(), res.d_outlier_val,
                               n_out * elem_sz, cudaMemcpyDeviceToHost));
        std::fwrite(val.data(), 1, n_out * elem_sz, fp);
    }
    {
        std::vector<uint8_t> sign(n_out);
        CUDA_CHECK(cudaMemcpy(sign.data(), res.d_outlier_sign,
                               n_out, cudaMemcpyDeviceToHost));
        std::fwrite(sign.data(), 1, n_out, fp);
    }

    std::fclose(fp);
}

// ============================================================================
// run_quantize — main quantization pipeline
// ============================================================================

template <typename Real>
static QuantResult run_quantize(Real* d_data, Real* d_orig,
                                const QuantConfig& cfg, QuantWorkspace& ws)
{
    QuantResult res;
    res.precision = std::is_same<Real, float>::value ? PrecFlag::F32 : PrecFlag::F64;

    const size_t total = size_t(cfg.nx) * cfg.ny * cfg.nz;
    res.total_elems = total;
    res.orig_elems  = total;
    res.num_words   = (total + 63) / 64;

    const bool is2d = (cfg.nz <= 1);

    // --- Padding ---
    const int pnx = wavelet::dwt3d_padded_dim(cfg.nx, cfg.levels);
    const int pny = wavelet::dwt3d_padded_dim(cfg.ny, cfg.levels);
    const int pnz = is2d ? cfg.nz : wavelet::dwt3d_padded_dim(cfg.nz, cfg.levels);
    const bool needs_pad = (pnx != cfg.nx || pny != cfg.ny || pnz != cfg.nz);
    const size_t pad_total = size_t(pnx) * pny * std::max(pnz, 1);

    if (needs_pad) {
        std::printf("  Padding: %dx%dx%d -> %dx%dx%d\n",
                    cfg.nx, cfg.ny, cfg.nz, pnx, pny, pnz);
    }

    // --- Step 1: Subtract mean (+ pad if needed, fused into one kernel) ---
    res.mean = gpu_compute_mean<Real>(d_data, total, ws);

    const bool need_outliers = (cfg.mode == CompMode::PWE);
    Real* d_work = d_data;
    Real* d_pad  = nullptr;

    if (needs_pad) {
        CUDA_CHECK(cudaMalloc(&d_pad, pad_total * sizeof(Real)));
        int grid = static_cast<int>((pad_total + QUANT_BLOCK - 1) / QUANT_BLOCK);
        const int onz = std::max(cfg.nz, 1);
        const int pnz_safe = std::max(pnz, 1);
        if (need_outliers) {
            kernel_reflect_pad_subtract_mean_copy<Real><<<grid, QUANT_BLOCK>>>(
                d_data, d_pad, d_orig,
                cfg.nx, cfg.ny, onz, pnx, pny, pnz_safe, res.mean);
        } else {
            kernel_reflect_pad_subtract_mean<Real><<<grid, QUANT_BLOCK>>>(
                d_data, d_pad,
                cfg.nx, cfg.ny, onz, pnx, pny, pnz_safe, res.mean);
        }
        CUDA_CHECK(cudaGetLastError());
        d_work = d_pad;
    } else {
        if (need_outliers) {
            int grid = static_cast<int>((total + QUANT_BLOCK - 1) / QUANT_BLOCK);
            kernel_subtract_mean_copy<Real><<<grid, QUANT_BLOCK>>>(
                d_data, d_orig, total, res.mean);
            CUDA_CHECK(cudaGetLastError());
        } else {
            int grid = static_cast<int>((total + QUANT_BLOCK - 1) / QUANT_BLOCK);
            kernel_subtract_mean<Real><<<grid, QUANT_BLOCK>>>(d_data, total, res.mean);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // --- Step 2: Forward DWT (overloaded: float or double) ---
    wavelet::dwt3d_forward(d_work, pnx, pny, std::max(pnz, 1),
                           cfg.levels, pnx, pny);

    // --- Step 3: Choose q ---
    switch (cfg.mode) {
    case CompMode::PWE:
        res.q = cfg.quality * 1.5;
        break;
    case CompMode::PSNR:
        res.q = choose_q_psnr(cfg.data_range, cfg.quality);
        break;
    }

    // --- Step 4: Choose integer width & quantize ---
    double max_abs = gpu_abs_max<Real>(d_work, pad_total, ws);
    res.coeff_type = choose_uint_type(max_abs, res.q);

    size_t elem_sz      = uint_elem_size(res.coeff_type);
    size_t pad_words    = (pad_total + 63) / 64;
    size_t sign_bytes_p = pad_words * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&res.d_coeffs, pad_total * elem_sz));
    CUDA_CHECK(cudaMalloc(&res.d_signs,  sign_bytes_p));
    CUDA_CHECK(cudaMemset(res.d_signs, 0, sign_bytes_p));

    double inv_q = 1.0 / res.q;
    dispatch_quantize<Real>(d_work, res.d_coeffs, res.d_signs, pad_total,
                            inv_q, res.coeff_type);

    res.total_elems  = pad_total;
    res.num_words    = pad_words;
    res.outlier_type = res.coeff_type;

    // PSNR mode: no outlier pipeline
    if (!need_outliers) {
        if (d_pad) CUDA_CHECK(cudaFree(d_pad));
        return res;
    }

    // --- Step 5 (PWE): Inverse quantize → IDWT → outlier detection ---

    dispatch_inv_quantize<Real>(res.d_coeffs, res.d_signs, d_work, pad_total,
                                res.q, res.coeff_type);

    wavelet::dwt3d_inverse(d_work, pnx, pny, std::max(pnz, 1),
                           cfg.levels, pnx, pny);

    Real* d_recon = nullptr;
    if (needs_pad) {
        CUDA_CHECK(cudaMalloc(&d_recon, total * sizeof(Real)));
        wavelet::dwt3d_crop(d_work, d_recon,
                            cfg.nx, cfg.ny, std::max(cfg.nz, 1),
                            pnx, pny, std::max(pnz, 1));
        CUDA_CHECK(cudaFree(d_pad));
    } else {
        d_recon = d_work;
    }

    // --- Step 6: Two-pass outlier compaction ---
    double tol_eff = res.q / 1.5;
    double inv_tol = 1.0 / tol_eff;
    int blocks_out = static_cast<int>((total + QUANT_BLOCK - 1) / QUANT_BLOCK);

    uint8_t* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, total * sizeof(uint8_t)));
    kernel_outlier_flag_u8<Real><<<blocks_out, QUANT_BLOCK>>>(
        d_orig, d_recon, d_flags, total, tol_eff);
    CUDA_CHECK(cudaGetLastError());

    uint32_t* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, total * sizeof(uint32_t)));

    if (total > size_t(INT_MAX)) {
        std::fprintf(stderr, "Error: total elements %zu exceeds INT_MAX for CUB scan\n", total);
        std::exit(EXIT_FAILURE);
    }

    U8ToU32 cast_op;
    cub::TransformInputIterator<uint32_t, U8ToU32, const uint8_t*>
        flags_iter(d_flags, cast_op);

    size_t scan_needed = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_needed, flags_iter, d_offsets,
        static_cast<int>(total)));
    ws.ensure_scan_temp(scan_needed);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_scan_temp, ws.scan_temp_bytes, flags_iter, d_offsets,
        static_cast<int>(total)));

    uint64_t* d_out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(uint64_t)));
    kernel_outlier_count_from_last_u8<<<1, 1>>>(d_offsets, d_flags, total,
                                                d_out_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&res.num_outliers, d_out_count, sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out_count));

    if (res.num_outliers > 0) {
        CUDA_CHECK(cudaMalloc(&res.d_outlier_idx,
                               res.num_outliers * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&res.d_outlier_val,
                               res.num_outliers * elem_sz));
        CUDA_CHECK(cudaMalloc(&res.d_outlier_sign,
                               res.num_outliers * sizeof(uint8_t)));
        dispatch_outlier_compact<Real>(d_orig, d_recon, d_flags, d_offsets,
                                       res.d_outlier_idx, res.d_outlier_val,
                                       res.d_outlier_sign,
                                       total, inv_tol, res.outlier_type);
    }

    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_offsets));
    if (needs_pad && d_recon) { CUDA_CHECK(cudaFree(d_recon)); }

    return res;
}

// ============================================================================
// main
// ============================================================================

// ============================================================================
// run_and_save — run pipeline for a given precision, print results, save files
// ============================================================================

template <typename Real>
static int run_and_save(const std::vector<Real>& h_data,
                        const QuantConfig& cfg, CompMode mode,
                        double data_range, const std::string& prefix)
{
    const size_t total = h_data.size();

    Real* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), total * sizeof(Real),
                           cudaMemcpyHostToDevice));

    const bool need_outliers = (mode == CompMode::PWE);
    Real* d_orig = nullptr;
    if (need_outliers)
        CUDA_CHECK(cudaMalloc(&d_orig, total * sizeof(Real)));

    QuantWorkspace ws;
    {
        const int pnx = wavelet::dwt3d_padded_dim(cfg.nx, cfg.levels);
        const int pny = wavelet::dwt3d_padded_dim(cfg.ny, cfg.levels);
        const int pnz = (cfg.nz <= 1) ? cfg.nz
                       : wavelet::dwt3d_padded_dim(cfg.nz, cfg.levels);
        size_t pad_total = size_t(pnx) * pny * std::max(pnz, 1);
        size_t max_n = std::max(total, pad_total);
        int max_blocks = static_cast<int>(
            (max_n + QUANT_BLOCK * 2 - 1) / (QUANT_BLOCK * 2));
        ws.alloc_reduce(max_blocks);
    }

    QuantResult res = run_quantize<Real>(d_data, d_orig, cfg, ws);

    std::printf("Results:\n");
    std::printf("  Precision:      %s\n",
                res.precision == PrecFlag::F32 ? "float" : "double");
    std::printf("  Mean:           %.15e\n", res.mean);
    std::printf("  q:              %.15e\n", res.q);
    std::printf("  Coeff type:     %s\n", uint_type_name(res.coeff_type));

    if (mode == CompMode::PSNR) {
        std::string combined_path = prefix + ".bin";
        save_combined(combined_path, res);

        std::printf("\nOutput file:\n");
        std::printf("  %s\n", combined_path.c_str());
    } else {
        std::string coeff_path   = prefix + "_coefficients.bin";
        std::string sign_path    = prefix + "_signs.bin";
        std::string outlier_path = prefix + "_outliers.bin";

        save_coefficients(coeff_path, res);
        save_signs(sign_path, res);
        save_outliers(outlier_path, res);

        std::printf("  Eff. tolerance: %.15e\n", res.q / 1.5);
        std::printf("  Outlier type:   %s\n", uint_type_name(res.outlier_type));
        std::printf("  Outliers:       %llu / %zu (%.4f%%)\n",
                    static_cast<unsigned long long>(res.num_outliers),
                    res.orig_elems,
                    100.0 * res.num_outliers / res.orig_elems);
        std::printf("\nOutput files:\n");
        std::printf("  %s\n", coeff_path.c_str());
        std::printf("  %s\n", sign_path.c_str());
        std::printf("  %s\n", outlier_path.c_str());
    }

    res.free_device();
    ws.free_all();
    CUDA_CHECK(cudaFree(d_data));
    if (d_orig) CUDA_CHECK(cudaFree(d_orig));

    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    CompMode mode = CompMode::PWE;
    int prec_arg = -1;  // -1 = auto-detect, 0 = double, 1 = float
    std::vector<char*> pos;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "psnr") == 0) mode = CompMode::PSNR;
        } else if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "float") == 0) prec_arg = 1;
            else prec_arg = 0;
        } else {
            pos.push_back(argv[i]);
        }
    }

    if (pos.size() < 5) {
        std::fprintf(stderr,
            "Usage: %s <input.raw> <nx> <ny> <nz> <quality> [levels] [prefix]\n"
            "       [--mode pwe|psnr] [--precision float|double]\n\n"
            "  quality:    tolerance (PWE) or target PSNR in dB (PSNR)\n"
            "  levels:     DWT depth (auto if omitted)\n"
            "  prefix:     output file prefix (default: temp/quant_out)\n"
            "  precision:  float or double (auto-detect from file size if omitted)\n",
            argv[0]);
        return 1;
    }

    const char* input_path = pos[0];
    const int nx      = std::atoi(pos[1]);
    const int ny      = std::atoi(pos[2]);
    const int nz      = std::atoi(pos[3]);
    const double quality = std::atof(pos[4]);
    const int levels  = (pos.size() > 5 && std::atoi(pos[5]) > 0)
                        ? std::atoi(pos[5])
                        : auto_dwt_levels(nx, ny, nz);
    const std::string prefix = (pos.size() > 6) ? pos[6] : "temp/quant_out";

    const size_t total = size_t(nx) * ny * nz;

    // --- Detect precision from file size or flag ---
    FILE* fp = std::fopen(input_path, "rb");
    if (!fp) { std::perror(input_path); return 1; }
    std::fseek(fp, 0, SEEK_END);
    long fsize = std::ftell(fp);
    std::rewind(fp);

    bool use_float;
    if (prec_arg >= 0) {
        use_float = (prec_arg == 1);
    } else {
        use_float = (fsize == static_cast<long>(total * sizeof(float)));
    }

    std::printf("Wavelet Quantizer\n");
    std::printf("  Input:     %s\n", input_path);
    std::printf("  Dims:      %d x %d x %d = %zu elements\n", nx, ny, nz, total);
    std::printf("  Mode:      %s\n", mode == CompMode::PWE ? "PWE" : "PSNR");
    std::printf("  Precision: %s\n", use_float ? "float" : "double");
    if (mode == CompMode::PWE)
        std::printf("  Tolerance: %.6e\n", quality);
    else
        std::printf("  Target:    %.2f dB\n", quality);
    std::printf("  Levels:    %d\n", levels);
    std::printf("  Prefix:    %s\n\n", prefix.c_str());

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n\n", prop.name);

    QuantConfig cfg;
    cfg.nx = nx;  cfg.ny = ny;  cfg.nz = nz;
    cfg.levels = levels;
    cfg.mode   = mode;
    cfg.quality = quality;

    if (use_float) {
        std::vector<float> h_data(total);
        if (fsize == static_cast<long>(total * sizeof(float))) {
            size_t nr = std::fread(h_data.data(), sizeof(float), total, fp);
            std::fclose(fp);
            if (nr != total) { std::fprintf(stderr, "Error: short read\n"); return 1; }
        } else {
            std::vector<double> h_tmp(total);
            size_t nr = std::fread(h_tmp.data(), sizeof(double), total, fp);
            std::fclose(fp);
            if (nr != total) { std::fprintf(stderr, "Error: short read\n"); return 1; }
            for (size_t i = 0; i < total; i++)
                h_data[i] = static_cast<float>(h_tmp[i]);
            std::printf("  (double input converted to float for processing)\n");
        }
        float fmin = *std::min_element(h_data.begin(), h_data.end());
        float fmax = *std::max_element(h_data.begin(), h_data.end());
        cfg.data_range = static_cast<double>(fmax) - static_cast<double>(fmin);
        if (mode == CompMode::PSNR)
            std::printf("  Data range: %.6e\n\n", cfg.data_range);
        return run_and_save<float>(h_data, cfg, mode, cfg.data_range, prefix);
    } else {
        std::vector<double> h_data(total);
        if (fsize == static_cast<long>(total * sizeof(float))) {
            std::vector<float> h_tmp(total);
            size_t nr = std::fread(h_tmp.data(), sizeof(float), total, fp);
            std::fclose(fp);
            if (nr != total) { std::fprintf(stderr, "Error: short read\n"); return 1; }
            for (size_t i = 0; i < total; i++)
                h_data[i] = static_cast<double>(h_tmp[i]);
            std::printf("  (float input promoted to double for processing)\n");
        } else {
            size_t nr = std::fread(h_data.data(), sizeof(double), total, fp);
            std::fclose(fp);
            if (nr != total) { std::fprintf(stderr, "Error: short read\n"); return 1; }
        }
        auto [it_min, it_max] = std::minmax_element(h_data.begin(), h_data.end());
        cfg.data_range = *it_max - *it_min;
        if (mode == CompMode::PSNR)
            std::printf("  Data range: %.6e\n\n", cfg.data_range);
        return run_and_save<double>(h_data, cfg, mode, cfg.data_range, prefix);
    }
}
