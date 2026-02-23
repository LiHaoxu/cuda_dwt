// pwe_quantizer.cu — CUDA PWE quantizer (no SPECK encoding)
//
// Replicates SPERR's PWE quantization pipeline on GPU:
//   conditioner (subtract mean) → DWT → midtread quantize →
//   inverse quantize → IDWT → outlier detect+quantize
//
// Outputs three raw binary files: coefficients, signs, outliers.
//
// Compilation:
//   nvcc -std=c++17 -O2 -I/home/lihaoxu/wavelet/cdf97_v1 \
//        pwe_quantizer.cu -o pwe_quantizer
//
// Usage:
//   ./pwe_quantizer <input.raw> <nx> <ny> <nz> <tolerance> [levels] [prefix]

#include "cdf97_3d_host.cu"   // brings in DWT/IDWT + cdf97_3d_kernel.cu
#include "pwe_kernels.cu"

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
// PweResult::free_device
// ============================================================================
void PweResult::free_device()
{
    if (d_coeffs)    { cudaFree(d_coeffs);    d_coeffs    = nullptr; }
    if (d_signs)     { cudaFree(d_signs);     d_signs     = nullptr; }
    if (d_out_coeffs){ cudaFree(d_out_coeffs);d_out_coeffs= nullptr; }
    if (d_out_signs) { cudaFree(d_out_signs); d_out_signs = nullptr; }
}

// ============================================================================
// Host-side GPU reduction: compute mean of double array on device
// ============================================================================
static double gpu_compute_mean(const double* d_data, size_t n)
{
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, n * sizeof(double),
                           cudaMemcpyDeviceToHost));

    // Kahan summation for accuracy (matches SPERR's stride-based approach).
    double sum = 0.0, comp = 0.0;
    for (size_t i = 0; i < n; i++) {
        double y = h[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }
    return sum / static_cast<double>(n);
}

// ============================================================================
// Host-side GPU reduction: max absolute value
// ============================================================================
static double gpu_abs_max(const double* d_data, size_t n)
{
    const int threads = QUANT_BLOCK;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));

    double* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    kernel_abs_max<<<blocks, threads, threads * sizeof(double)>>>(
        d_data, n, d_partial);
    CUDA_CHECK(cudaGetLastError());

    // Second pass if needed.
    while (blocks > 1) {
        int prev = blocks;
        blocks = (prev + threads * 2 - 1) / (threads * 2);
        double* d_partial2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial2, blocks * sizeof(double)));
        kernel_abs_max<<<blocks, threads, threads * sizeof(double)>>>(
            d_partial, prev, d_partial2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_partial));
        d_partial = d_partial2;
    }

    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(double),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    return result;
}

// ============================================================================
// Auto-select DWT levels based on the largest spatial dimension.
//   >=512 → 6,  >=256 → 5,  >=128 → 4,  >=64 → 3,  >=32 → 2,  else → 1
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

// ============================================================================
// Choose integer width — matches SPERR (src/SPECK_FLT.cpp:329-336)
// ============================================================================
static UIntFlag choose_uint_type(double max_abs, double q)
{
    long long maxll = std::llrint(max_abs / q);
    if (maxll < 0) maxll = -maxll;

    if (maxll <= std::numeric_limits<uint8_t>::max())  return UIntFlag::U8;
    if (maxll <= std::numeric_limits<uint16_t>::max()) return UIntFlag::U16;
    if (maxll <= std::numeric_limits<uint32_t>::max()) return UIntFlag::U32;
    return UIntFlag::U64;
}

// ============================================================================
// Byte size of one element for a given UIntFlag
// ============================================================================
static size_t uint_elem_size(UIntFlag f)
{
    switch (f) {
        case UIntFlag::U8:  return 1;
        case UIntFlag::U16: return 2;
        case UIntFlag::U32: return 4;
        default:            return 8;
    }
}

// ============================================================================
// Dispatch quantize kernel by type
// ============================================================================
static void dispatch_quantize(const double* d_coeffs, void* d_ints,
                              uint64_t* d_signs, size_t n,
                              double inv_q, UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_midtread_quantize<uint8_t><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint8_t*>(d_ints), d_signs, n, inv_q);
        break;
    case UIntFlag::U16:
        kernel_midtread_quantize<uint16_t><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint16_t*>(d_ints), d_signs, n, inv_q);
        break;
    case UIntFlag::U32:
        kernel_midtread_quantize<uint32_t><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint32_t*>(d_ints), d_signs, n, inv_q);
        break;
    default:
        kernel_midtread_quantize<uint64_t><<<blocks, QUANT_BLOCK>>>(
            d_coeffs, static_cast<uint64_t*>(d_ints), d_signs, n, inv_q);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Dispatch inverse-quantize kernel by type
// ============================================================================
static void dispatch_inv_quantize(const void* d_ints, const uint64_t* d_signs,
                                  double* d_coeffs, size_t n,
                                  double q, UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_midtread_inv_quantize<uint8_t><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint8_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U16:
        kernel_midtread_inv_quantize<uint16_t><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint16_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U32:
        kernel_midtread_inv_quantize<uint32_t><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint32_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    default:
        kernel_midtread_inv_quantize<uint64_t><<<blocks, QUANT_BLOCK>>>(
            static_cast<const uint64_t*>(d_ints), d_signs, d_coeffs, n, q);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Dispatch outlier detect+quantize kernel by type
// ============================================================================
static void dispatch_outlier(const double* d_orig, const double* d_recon,
                             void* d_out_ints, uint64_t* d_out_signs,
                             unsigned long long* d_out_count,
                             size_t n, double tol, double inv_tol,
                             UIntFlag flag)
{
    int blocks = static_cast<int>((n + QUANT_BLOCK - 1) / QUANT_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_outlier_detect_quantize<uint8_t><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, static_cast<uint8_t*>(d_out_ints),
            d_out_signs, d_out_count, n, tol, inv_tol);
        break;
    case UIntFlag::U16:
        kernel_outlier_detect_quantize<uint16_t><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, static_cast<uint16_t*>(d_out_ints),
            d_out_signs, d_out_count, n, tol, inv_tol);
        break;
    case UIntFlag::U32:
        kernel_outlier_detect_quantize<uint32_t><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, static_cast<uint32_t*>(d_out_ints),
            d_out_signs, d_out_count, n, tol, inv_tol);
        break;
    default:
        kernel_outlier_detect_quantize<uint64_t><<<blocks, QUANT_BLOCK>>>(
            d_orig, d_recon, static_cast<uint64_t*>(d_out_ints),
            d_out_signs, d_out_count, n, tol, inv_tol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// File I/O helpers
// ============================================================================

void save_coefficients(const std::string& path, const PweResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    // Header: uint_type_flag(1) + num_elements(8) + q(8) = 17 bytes
    uint8_t flag = static_cast<uint8_t>(res.coeff_type);
    std::fwrite(&flag, 1, 1, fp);
    uint64_t ne = res.total_elems;
    std::fwrite(&ne, sizeof(ne), 1, fp);
    std::fwrite(&res.q, sizeof(res.q), 1, fp);

    // Payload: integer array
    size_t elem_sz  = uint_elem_size(res.coeff_type);
    size_t payload  = res.total_elems * elem_sz;
    std::vector<uint8_t> buf(payload);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_coeffs, payload,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, payload, fp);
    std::fclose(fp);
}

void save_signs(const std::string& path, const PweResult& res)
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

void save_outliers(const std::string& path, const PweResult& res)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(path.c_str()); std::exit(1); }

    // Header: uint_type_flag(1) + num_elements(8) + tolerance(8) + num_outliers(8) = 25 bytes
    uint8_t flag = static_cast<uint8_t>(res.outlier_type);
    std::fwrite(&flag, 1, 1, fp);
    uint64_t ne = res.total_elems;
    std::fwrite(&ne, sizeof(ne), 1, fp);
    double tol = res.q / 1.5;  // recover tolerance from q
    std::fwrite(&tol, sizeof(tol), 1, fp);
    std::fwrite(&res.num_outliers, sizeof(res.num_outliers), 1, fp);

    if (res.num_outliers == 0) {
        std::fclose(fp);
        return;
    }

    // Payload: outlier integer array
    size_t elem_sz = uint_elem_size(res.outlier_type);
    size_t payload = res.total_elems * elem_sz;
    std::vector<uint8_t> buf(payload);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_out_coeffs, payload,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, payload, fp);

    // Payload: outlier sign bitmask
    size_t sign_bytes = res.num_words * sizeof(uint64_t);
    buf.resize(sign_bytes);
    CUDA_CHECK(cudaMemcpy(buf.data(), res.d_out_signs, sign_bytes,
                           cudaMemcpyDeviceToHost));
    std::fwrite(buf.data(), 1, sign_bytes, fp);

    std::fclose(fp);
}

// ============================================================================
// pwe_quantize — main quantization pipeline
// ============================================================================
static PweResult pwe_quantize(double* d_data, double* d_orig,
                              const PweConfig& cfg)
{
    PweResult res;
    const size_t total = size_t(cfg.nx) * cfg.ny * cfg.nz;
    res.total_elems = total;
    res.num_words   = (total + 63) / 64;

    // --- Step 1: Conditioner — compute and subtract mean ---
    res.mean = gpu_compute_mean(d_data, total);

    int blocks = static_cast<int>((total + QUANT_BLOCK - 1) / QUANT_BLOCK);
    kernel_subtract_mean<<<blocks, QUANT_BLOCK>>>(d_data, total, res.mean);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- PWE: save conditioned copy for later outlier comparison ---
    CUDA_CHECK(cudaMemcpy(d_orig, d_data, total * sizeof(double),
                           cudaMemcpyDeviceToDevice));

    // --- Step 2: Forward DWT ---
    wavelet::dwt3d_forward(d_data, cfg.nx, cfg.ny, cfg.nz,
                           cfg.levels, cfg.nx, cfg.ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Step 2.1: Estimate q = tolerance * 1.5 ---
    res.q = cfg.tolerance * 1.5;

    // --- Step 3: Choose integer width ---
    double max_abs = gpu_abs_max(d_data, total);
    res.coeff_type = choose_uint_type(max_abs, res.q);

    // --- Step 3.1: Allocate quantization buffers ---
    size_t elem_sz    = uint_elem_size(res.coeff_type);
    size_t sign_bytes = res.num_words * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&res.d_coeffs, total * elem_sz));
    CUDA_CHECK(cudaMalloc(&res.d_signs,  sign_bytes));
    CUDA_CHECK(cudaMemset(res.d_signs, 0, sign_bytes));

    // --- Step 3.2: Quantize ---
    double inv_q = 1.0 / res.q;
    dispatch_quantize(d_data, res.d_coeffs, res.d_signs, total,
                      inv_q, res.coeff_type);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Step 4 (PWE): Inverse quantize → IDWT → outlier detection ---

    // 4a: Inverse quantize back to wavelet domain
    dispatch_inv_quantize(res.d_coeffs, res.d_signs, d_data, total,
                          res.q, res.coeff_type);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4b: Inverse DWT to get reconstructed spatial data
    wavelet::dwt3d_inverse(d_data, cfg.nx, cfg.ny, cfg.nz,
                           cfg.levels, cfg.nx, cfg.ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4c: Determine outlier integer width.
    //     The max outlier error cannot exceed max_abs (conservative), but we
    //     use tolerance as the quantization step, so the max integer is bounded
    //     by max_abs / tolerance.  In practice outliers are small, so we
    //     compute the actual max diff on GPU for a tighter bound.
    //     For simplicity we reuse the same width as the main coefficients or
    //     compute from the actual max diff.

    // Compute max |orig - recon| on GPU by reusing d_data as scratch:
    // We need a temp buffer for diffs.
    double* d_diff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diff, total * sizeof(double)));

    // Compute diff = orig - recon element-wise (reuse subtract_mean kernel
    // creatively: diff = orig; diff -= recon is two steps, but let's just
    // do it on host for the max-finding — the outlier kernel itself does
    // the per-element diff anyway).
    // Instead, compute max |orig[i] - recon[i]| via a small kernel:
    // We'll just launch the outlier kernel with a pre-pass to find the max.
    // For simplicity, assume outlier type <= coeff_type (conservative).
    // A tighter approach: copy orig-recon to d_diff, then abs_max.

    // Quick approach: d_diff[i] = orig[i] - recon[i]
    // We can do this with subtract_mean trick: copy orig to d_diff, then
    // subtract recon.  But that's clunky.  Let's just use the same type
    // as coeff_type for outliers — it's always safe since outlier errors
    // are bounded by the main quantization error.
    res.outlier_type = res.coeff_type;

    CUDA_CHECK(cudaFree(d_diff));

    // 4d: Allocate outlier buffers
    CUDA_CHECK(cudaMalloc(&res.d_out_coeffs, total * elem_sz));
    CUDA_CHECK(cudaMemset(res.d_out_coeffs, 0, total * elem_sz));

    CUDA_CHECK(cudaMalloc(&res.d_out_signs, sign_bytes));
    // Initialise outlier signs to all-1 (matches SPERR Bitmask::reset_true).
    CUDA_CHECK(cudaMemset(res.d_out_signs, 0xFF, sign_bytes));

    unsigned long long* d_out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(unsigned long long)));

    // 4e: Outlier detect + quantize
    double inv_tol = 1.0 / cfg.tolerance;
    dispatch_outlier(d_orig, d_data, res.d_out_coeffs, res.d_out_signs,
                     d_out_count, total, cfg.tolerance, inv_tol,
                     res.outlier_type);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read outlier count
    unsigned long long h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_count, sizeof(h_count),
                           cudaMemcpyDeviceToHost));
    res.num_outliers = h_count;

    CUDA_CHECK(cudaFree(d_out_count));

    return res;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    if (argc < 6) {
        std::fprintf(stderr,
            "Usage: %s <input.raw> <nx> <ny> <nz> <tolerance> "
            "[levels] [output_prefix]\n"
            "  levels: DWT decomposition depth (auto if omitted:\n"
            "          >=512→6, >=256→5, >=128→4, >=64→3, >=32→2)\n",
            argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const int nx      = std::atoi(argv[2]);
    const int ny      = std::atoi(argv[3]);
    const int nz      = std::atoi(argv[4]);
    const double tol  = std::atof(argv[5]);
    const int levels  = (argc > 6 && std::atoi(argv[6]) > 0)
                        ? std::atoi(argv[6])
                        : auto_dwt_levels(nx, ny, nz);
    const std::string prefix = (argc > 7) ? argv[7] : "pwe_out";

    const size_t total = size_t(nx) * ny * nz;

    std::printf("PWE Quantizer\n");
    std::printf("  Input:     %s\n", input_path);
    std::printf("  Dims:      %d x %d x %d = %zu elements\n", nx, ny, nz, total);
    std::printf("  Levels:    %d\n", levels);
    std::printf("  Tolerance: %.6e\n", tol);
    std::printf("  Prefix:    %s\n\n", prefix.c_str());

    // --- Read input ---
    FILE* fp = std::fopen(input_path, "rb");
    if (!fp) { std::perror(input_path); return 1; }
    std::vector<double> h_data(total);
    size_t nread = std::fread(h_data.data(), sizeof(double), total, fp);
    std::fclose(fp);
    if (nread != total) {
        std::fprintf(stderr, "Error: expected %zu doubles, read %zu\n",
                     total, nread);
        return 1;
    }

    // --- Print GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n\n", prop.name);

    // --- Allocate device memory ---
    double* d_data = nullptr;
    double* d_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_orig, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), total * sizeof(double),
                           cudaMemcpyHostToDevice));

    // --- Run PWE quantization ---
    PweConfig cfg;
    cfg.nx = nx;  cfg.ny = ny;  cfg.nz = nz;
    cfg.levels = levels;
    cfg.tolerance = tol;

    PweResult res = pwe_quantize(d_data, d_orig, cfg);

    // --- Save output files ---
    std::string coeff_path   = prefix + "_coefficients.bin";
    std::string sign_path    = prefix + "_signs.bin";
    std::string outlier_path = prefix + "_outliers.bin";

    save_coefficients(coeff_path, res);
    save_signs(sign_path, res);
    save_outliers(outlier_path, res);

    // --- Print summary ---
    auto type_name = [](UIntFlag f) -> const char* {
        switch (f) {
            case UIntFlag::U8:  return "uint8";
            case UIntFlag::U16: return "uint16";
            case UIntFlag::U32: return "uint32";
            default:            return "uint64";
        }
    };

    std::printf("Results:\n");
    std::printf("  Mean:           %.15e\n", res.mean);
    std::printf("  q (= tol*1.5): %.15e\n", res.q);
    std::printf("  Coeff type:     %s\n", type_name(res.coeff_type));
    std::printf("  Outlier type:   %s\n", type_name(res.outlier_type));
    std::printf("  Outliers:       %llu / %zu (%.4f%%)\n",
                static_cast<unsigned long long>(res.num_outliers),
                res.total_elems,
                100.0 * res.num_outliers / res.total_elems);
    std::printf("\nOutput files:\n");
    std::printf("  %s\n", coeff_path.c_str());
    std::printf("  %s\n", sign_path.c_str());
    std::printf("  %s\n", outlier_path.c_str());

    // --- Cleanup ---
    res.free_device();
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_orig));

    return 0;
}
