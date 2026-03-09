// decoder.cu — Reconstruct original data from quantizer output files
//
// Inverse pipeline:
//   read coefficients + signs → inv-quantize → IDWT →
//   crop + add mean (fused) → [apply outlier corrections] → write output
//
// Compilation:
//   cd /home/lihaoxu/wavelet
//   nvcc -std=c++17 -O2 -I include src/decoder.cu -o decoder
//
// Usage:
//   ./decoder <prefix> <nx> <ny> <nz> [levels] [output.raw]
//
// PWE mode (auto-detected when outlier file exists):
//   reads <prefix>_coefficients.bin, <prefix>_signs.bin, <prefix>_outliers.bin
//
// PSNR mode (single combined file):
//   reads <prefix>.bin

#include "cdf97_3d_host.cu"
#include "cdf97_3d_f32_host.cu"
#include "quant_kernels.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

static constexpr int DEC_BLOCK = 256;

// ============================================================================
// Fused crop + add_mean kernel — templated on Real
// ============================================================================

template <typename Real>
__global__ void kernel_crop_add_mean(
    const Real* __restrict__ src, Real* __restrict__ dst,
    int orig_nx, int orig_ny, int orig_nz,
    int pad_nx,  int pad_ny,  int pad_nz,
    double mean)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t tot = size_t(orig_nx) * orig_ny * orig_nz;
    if (idx >= tot) return;

    int x = static_cast<int>(idx % orig_nx);
    int y = static_cast<int>((idx / orig_nx) % orig_ny);
    int z = static_cast<int>(idx / (size_t(orig_nx) * orig_ny));

    double v = static_cast<double>(src[size_t(z) * pad_nx * pad_ny + size_t(y) * pad_nx + x]);
    dst[idx] = static_cast<Real>(v + mean);
}

// ============================================================================
// Utilities (same as encoder)
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

static const char* uint_type_name(UIntFlag f)
{
    switch (f) {
        case UIntFlag::U8:  return "uint8";
        case UIntFlag::U16: return "uint16";
        case UIntFlag::U32: return "uint32";
        default:            return "uint64";
    }
}

// ============================================================================
// Templated dispatch helpers
// ============================================================================

template <typename Real>
static void dispatch_inv_quantize(const void* d_ints, const uint64_t* d_signs,
                                  Real* d_coeffs, size_t n,
                                  double q, UIntFlag flag)
{
    int blocks = static_cast<int>((n + DEC_BLOCK - 1) / DEC_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_midtread_inv_quantize<uint8_t, Real><<<blocks, DEC_BLOCK>>>(
            static_cast<const uint8_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U16:
        kernel_midtread_inv_quantize<uint16_t, Real><<<blocks, DEC_BLOCK>>>(
            static_cast<const uint16_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    case UIntFlag::U32:
        kernel_midtread_inv_quantize<uint32_t, Real><<<blocks, DEC_BLOCK>>>(
            static_cast<const uint32_t*>(d_ints), d_signs, d_coeffs, n, q);
        break;
    default:
        kernel_midtread_inv_quantize<uint64_t, Real><<<blocks, DEC_BLOCK>>>(
            static_cast<const uint64_t*>(d_ints), d_signs, d_coeffs, n, q);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Real>
static void dispatch_apply_outliers(Real* d_data,
                                    const uint32_t* d_idx, const void* d_val,
                                    const uint8_t* d_sign,
                                    uint64_t n_outliers, double tol, UIntFlag flag)
{
    if (n_outliers == 0) return;
    int blocks = static_cast<int>((n_outliers + DEC_BLOCK - 1) / DEC_BLOCK);
    switch (flag) {
    case UIntFlag::U8:
        kernel_apply_outliers<uint8_t, Real><<<blocks, DEC_BLOCK>>>(
            d_data, d_idx, static_cast<const uint8_t*>(d_val), d_sign, n_outliers, tol);
        break;
    case UIntFlag::U16:
        kernel_apply_outliers<uint16_t, Real><<<blocks, DEC_BLOCK>>>(
            d_data, d_idx, static_cast<const uint16_t*>(d_val), d_sign, n_outliers, tol);
        break;
    case UIntFlag::U32:
        kernel_apply_outliers<uint32_t, Real><<<blocks, DEC_BLOCK>>>(
            d_data, d_idx, static_cast<const uint32_t*>(d_val), d_sign, n_outliers, tol);
        break;
    default:
        kernel_apply_outliers<uint64_t, Real><<<blocks, DEC_BLOCK>>>(
            d_data, d_idx, static_cast<const uint64_t*>(d_val), d_sign, n_outliers, tol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// File header structures (26 bytes: 25-byte core + 1-byte precision)
// ============================================================================

struct CoeffHeader {
    UIntFlag type;
    uint64_t num_elements;
    double   q;
    double   mean;
    PrecFlag precision = PrecFlag::F64;
};

static CoeffHeader read_coeff_header(FILE* fp, long fsize)
{
    CoeffHeader h;
    uint8_t flag;
    if (std::fread(&flag, 1, 1, fp) != 1 ||
        std::fread(&h.num_elements, sizeof(h.num_elements), 1, fp) != 1 ||
        std::fread(&h.q, sizeof(h.q), 1, fp) != 1 ||
        std::fread(&h.mean, sizeof(h.mean), 1, fp) != 1) {
        std::fputs("header read error\n", stderr); std::exit(1);
    }
    h.type = static_cast<UIntFlag>(flag);

    // Try reading precision byte (26-byte header).
    // If only 25-byte core header, the remaining bytes are all payload.
    long pos = std::ftell(fp);
    size_t elem_sz = uint_elem_size(h.type);
    size_t payload_core = h.num_elements * elem_sz;
    long remaining = fsize - pos;

    size_t sign_words = (h.num_elements + 63) / 64;
    size_t combined_payload = payload_core + sign_words * sizeof(uint64_t);

    if (remaining == static_cast<long>(payload_core + 1) ||
        remaining == static_cast<long>(combined_payload + 1)) {
        uint8_t prec;
        if (std::fread(&prec, 1, 1, fp) == 1 && prec <= 1)
            h.precision = static_cast<PrecFlag>(prec);
    }

    return h;
}

struct OutlierHeader {
    UIntFlag type;
    uint64_t orig_elements;
    double   tolerance;
    uint64_t num_outliers;
    PrecFlag precision = PrecFlag::F64;
};

static OutlierHeader read_outlier_header(FILE* fp, long fsize)
{
    OutlierHeader h;
    uint8_t flag;
    if (std::fread(&flag, 1, 1, fp) != 1 ||
        std::fread(&h.orig_elements, sizeof(h.orig_elements), 1, fp) != 1 ||
        std::fread(&h.tolerance, sizeof(h.tolerance), 1, fp) != 1 ||
        std::fread(&h.num_outliers, sizeof(h.num_outliers), 1, fp) != 1) {
        std::fputs("outlier header read error\n", stderr); std::exit(1);
    }
    h.type = static_cast<UIntFlag>(flag);

    long pos = std::ftell(fp);
    size_t out_elem_sz = uint_elem_size(h.type);
    size_t sparse_payload = h.num_outliers * (sizeof(uint32_t) + out_elem_sz + 1);
    long remaining = fsize - pos;

    if (remaining == static_cast<long>(sparse_payload + 1)) {
        uint8_t prec;
        if (std::fread(&prec, 1, 1, fp) == 1 && prec <= 1)
            h.precision = static_cast<PrecFlag>(prec);
    }

    return h;
}

// ============================================================================
// run_decode<Real> — main reconstruction pipeline
// ============================================================================

template <typename Real>
static int run_decode(const CoeffHeader& ch,
                      const std::vector<uint8_t>& h_ints,
                      const std::vector<uint8_t>& h_signs,
                      bool is_pwe,
                      const std::string& outlier_path,
                      int nx, int ny, int nz, int levels,
                      const std::string& output_path)
{
    const size_t total = size_t(nx) * ny * nz;
    const bool is2d = (nz <= 1);
    const int pnx = wavelet::dwt3d_padded_dim(nx, levels);
    const int pny = wavelet::dwt3d_padded_dim(ny, levels);
    const int pnz = is2d ? nz : wavelet::dwt3d_padded_dim(nz, levels);
    const bool needs_pad = (pnx != nx || pny != ny || pnz != nz);
    const size_t pad_total = size_t(pnx) * pny * std::max(pnz, 1);

    size_t elem_sz = uint_elem_size(ch.type);

    // ---- Step 2: Upload and inverse-quantize ----
    void* d_ints = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ints, pad_total * elem_sz));
    CUDA_CHECK(cudaMemcpy(d_ints, h_ints.data(), pad_total * elem_sz,
                           cudaMemcpyHostToDevice));

    uint64_t* d_signs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signs, h_signs.size()));
    CUDA_CHECK(cudaMemcpy(d_signs, h_signs.data(), h_signs.size(),
                           cudaMemcpyHostToDevice));

    Real* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, pad_total * sizeof(Real)));

    dispatch_inv_quantize<Real>(d_ints, d_signs, d_work, pad_total, ch.q, ch.type);

    CUDA_CHECK(cudaFree(d_ints));
    CUDA_CHECK(cudaFree(d_signs));

    // ---- Step 3: Inverse DWT (overloaded on Real) ----
    wavelet::dwt3d_inverse(d_work, pnx, pny, std::max(pnz, 1),
                           levels, pnx, pny);

    // ---- Step 4: Crop + add mean ----
    Real* d_result = nullptr;
    if (needs_pad) {
        CUDA_CHECK(cudaMalloc(&d_result, total * sizeof(Real)));
        int grid = static_cast<int>((total + DEC_BLOCK - 1) / DEC_BLOCK);
        kernel_crop_add_mean<Real><<<grid, DEC_BLOCK>>>(
            d_work, d_result,
            nx, ny, std::max(nz, 1),
            pnx, pny, std::max(pnz, 1), ch.mean);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_work));
    } else {
        d_result = d_work;
        int grid = static_cast<int>((total + DEC_BLOCK - 1) / DEC_BLOCK);
        kernel_add_mean<Real><<<grid, DEC_BLOCK>>>(d_result, total, ch.mean);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 5 (PWE): Apply outlier corrections ----
    if (is_pwe) {
        FILE* fp_o = std::fopen(outlier_path.c_str(), "rb");
        if (!fp_o) { std::perror(outlier_path.c_str()); return 1; }
        std::fseek(fp_o, 0, SEEK_END);
        long osize = std::ftell(fp_o);
        std::rewind(fp_o);
        OutlierHeader oh = read_outlier_header(fp_o, osize);

        std::printf("  Tolerance:  %.15e\n", oh.tolerance);
        std::printf("  Outliers:   %llu / %llu (%.4f%%)\n",
                    static_cast<unsigned long long>(oh.num_outliers),
                    static_cast<unsigned long long>(oh.orig_elements),
                    oh.orig_elements > 0
                        ? 100.0 * oh.num_outliers / oh.orig_elements
                        : 0.0);

        if (oh.num_outliers > 0) {
            size_t out_elem_sz = uint_elem_size(oh.type);
            uint64_t n_out = oh.num_outliers;

            std::vector<uint32_t> h_idx(n_out);
            if (std::fread(h_idx.data(), sizeof(uint32_t), n_out, fp_o) != n_out) {
                std::fputs("read error\n", stderr); return 1;
            }
            std::vector<uint8_t> h_val(n_out * out_elem_sz);
            if (std::fread(h_val.data(), 1, n_out * out_elem_sz, fp_o) != n_out * out_elem_sz) {
                std::fputs("read error\n", stderr); return 1;
            }
            std::vector<uint8_t> h_sign(n_out);
            if (std::fread(h_sign.data(), 1, n_out, fp_o) != n_out) {
                std::fputs("read error\n", stderr); return 1;
            }
            std::fclose(fp_o);

            uint32_t* d_idx = nullptr;
            void*     d_val = nullptr;
            uint8_t*  d_sign = nullptr;
            CUDA_CHECK(cudaMalloc(&d_idx,  n_out * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_val,  n_out * out_elem_sz));
            CUDA_CHECK(cudaMalloc(&d_sign, n_out * sizeof(uint8_t)));
            CUDA_CHECK(cudaMemcpy(d_idx,  h_idx.data(),  n_out * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_val,  h_val.data(),  n_out * out_elem_sz,      cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sign, h_sign.data(), n_out * sizeof(uint8_t),  cudaMemcpyHostToDevice));

            dispatch_apply_outliers<Real>(d_result, d_idx, d_val, d_sign,
                                          n_out, oh.tolerance, oh.type);

            CUDA_CHECK(cudaFree(d_idx));
            CUDA_CHECK(cudaFree(d_val));
            CUDA_CHECK(cudaFree(d_sign));
        } else {
            std::fclose(fp_o);
        }
    }

    // ---- Step 6: Download and write output ----
    const char* suffix = std::is_same<Real, float>::value ? ".f32" : ".d64";
    std::string final_path = output_path;
    if (final_path.find(".f32") == std::string::npos &&
        final_path.find(".d64") == std::string::npos) {
        final_path = output_path.substr(0, output_path.rfind('.')) + suffix;
    }

    std::vector<Real> h_result(total);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, total * sizeof(Real),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    FILE* fp_out = std::fopen(final_path.c_str(), "wb");
    if (!fp_out) { std::perror(final_path.c_str()); return 1; }
    std::fwrite(h_result.data(), sizeof(Real), total, fp_out);
    std::fclose(fp_out);

    std::printf("\nReconstructed %zu %s → %s\n", total,
                std::is_same<Real, float>::value ? "floats" : "doubles",
                final_path.c_str());

    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: %s <prefix> <nx> <ny> <nz> [levels] [output.raw]\n\n"
            "  Reconstructs data from quantizer output files.\n"
            "  Precision auto-detected from file header.\n"
            "  PWE mode:  reads <prefix>_coefficients.bin, _signs.bin, _outliers.bin\n"
            "  PSNR mode: reads <prefix>.bin (auto-detected)\n",
            argv[0]);
        return 1;
    }

    const std::string prefix = argv[1];
    const int nx = std::atoi(argv[2]);
    const int ny = std::atoi(argv[3]);
    const int nz = std::atoi(argv[4]);
    const int levels = (argc > 5 && std::atoi(argv[5]) > 0)
                       ? std::atoi(argv[5])
                       : auto_dwt_levels(nx, ny, nz);
    const std::string output_path = (argc > 6) ? argv[6]
                                               : (prefix + "_reconstructed.d64");

    const size_t total = size_t(nx) * ny * nz;
    const bool is2d = (nz <= 1);

    const int pnx = wavelet::dwt3d_padded_dim(nx, levels);
    const int pny = wavelet::dwt3d_padded_dim(ny, levels);
    const int pnz = is2d ? nz : wavelet::dwt3d_padded_dim(nz, levels);
    const bool needs_pad = (pnx != nx || pny != ny || pnz != nz);
    const size_t pad_total = size_t(pnx) * pny * std::max(pnz, 1);

    // --- Detect mode ---
    std::string coeff_path   = prefix + "_coefficients.bin";
    std::string sign_path    = prefix + "_signs.bin";
    std::string outlier_path = prefix + "_outliers.bin";
    std::string combined_path = prefix + ".bin";

    FILE* fp_test = std::fopen(coeff_path.c_str(), "rb");
    const bool is_pwe = (fp_test != nullptr);
    if (fp_test) std::fclose(fp_test);

    std::printf("Wavelet Decoder\n");
    std::printf("  Prefix:    %s\n", prefix.c_str());
    std::printf("  Dims:      %d x %d x %d = %zu elements\n", nx, ny, nz, total);
    if (needs_pad)
        std::printf("  Padded:    %d x %d x %d = %zu elements\n", pnx, pny, pnz, pad_total);
    std::printf("  Levels:    %d\n", levels);
    std::printf("  Mode:      %s\n", is_pwe ? "PWE" : "PSNR");
    std::printf("  Output:    %s\n\n", output_path.c_str());

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n\n", prop.name);

    // ====================================================================
    // Step 1: Read header + payload
    // ====================================================================

    CoeffHeader ch;
    std::vector<uint8_t> h_ints;
    std::vector<uint8_t> h_signs;

    if (is_pwe) {
        FILE* fp_c = std::fopen(coeff_path.c_str(), "rb");
        if (!fp_c) { std::perror(coeff_path.c_str()); return 1; }
        std::fseek(fp_c, 0, SEEK_END);
        long csize = std::ftell(fp_c);
        std::rewind(fp_c);
        ch = read_coeff_header(fp_c, csize);

        size_t elem_sz = uint_elem_size(ch.type);
        size_t payload = ch.num_elements * elem_sz;
        h_ints.resize(payload);
        if (std::fread(h_ints.data(), 1, payload, fp_c) != payload) {
            std::fputs("read error\n", stderr); return 1;
        }
        std::fclose(fp_c);

        FILE* fp_s = std::fopen(sign_path.c_str(), "rb");
        if (!fp_s) { std::perror(sign_path.c_str()); return 1; }
        size_t sign_words = (ch.num_elements + 63) / 64;
        size_t sign_bytes = sign_words * sizeof(uint64_t);
        h_signs.resize(sign_bytes);
        if (std::fread(h_signs.data(), 1, sign_bytes, fp_s) != sign_bytes) {
            std::fputs("read error\n", stderr); return 1;
        }
        std::fclose(fp_s);
    } else {
        FILE* fp_cb = std::fopen(combined_path.c_str(), "rb");
        if (!fp_cb) { std::perror(combined_path.c_str()); return 1; }
        std::fseek(fp_cb, 0, SEEK_END);
        long cbsize = std::ftell(fp_cb);
        std::rewind(fp_cb);
        ch = read_coeff_header(fp_cb, cbsize);

        size_t elem_sz = uint_elem_size(ch.type);
        size_t payload = ch.num_elements * elem_sz;
        h_ints.resize(payload);
        if (std::fread(h_ints.data(), 1, payload, fp_cb) != payload) {
            std::fputs("read error\n", stderr); return 1;
        }

        size_t sign_words = (ch.num_elements + 63) / 64;
        size_t sign_bytes = sign_words * sizeof(uint64_t);
        h_signs.resize(sign_bytes);
        if (std::fread(h_signs.data(), 1, sign_bytes, fp_cb) != sign_bytes) {
            std::fputs("read error\n", stderr); return 1;
        }
        std::fclose(fp_cb);
    }

    if (ch.num_elements != pad_total) {
        std::fprintf(stderr,
            "Error: file has %llu elements but padded dims give %zu\n",
            static_cast<unsigned long long>(ch.num_elements), pad_total);
        return 1;
    }

    std::printf("  Precision:  %s\n", ch.precision == PrecFlag::F32 ? "float" : "double");
    std::printf("  Coeff type: %s\n", uint_type_name(ch.type));
    std::printf("  q:          %.15e\n", ch.q);
    std::printf("  Mean:       %.15e\n", ch.mean);

    // ====================================================================
    // Dispatch based on precision
    // ====================================================================

    if (ch.precision == PrecFlag::F32) {
        return run_decode<float>(ch, h_ints, h_signs, is_pwe, outlier_path,
                                 nx, ny, nz, levels, output_path);
    } else {
        return run_decode<double>(ch, h_ints, h_signs, is_pwe, outlier_path,
                                  nx, ny, nz, levels, output_path);
    }
}
