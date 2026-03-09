#ifndef WAVELET_QUANTIZER_CUH_
#define WAVELET_QUANTIZER_CUH_

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>

// Integer width flag — matches SPERR UINTType ordering.
enum class UIntFlag : uint8_t {
    U8  = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3
};

enum class CompMode : uint8_t { PWE = 0, PSNR = 1 };

// Precision flag for file header
enum class PrecFlag : uint8_t { F64 = 0, F32 = 1 };

struct QuantConfig {
    int      nx         = 0;
    int      ny         = 0;
    int      nz         = 0;
    int      levels     = 0;
    CompMode mode       = CompMode::PWE;
    double   quality    = 0.0;   // PWE: tolerance, PSNR: target PSNR (dB)
    double   data_range = 0.0;   // max - min of input (peak-to-peak, for PSNR)
};

struct QuantResult {
    void*     d_coeffs       = nullptr;
    uint64_t* d_signs        = nullptr;

    uint32_t* d_outlier_idx  = nullptr;
    void*     d_outlier_val  = nullptr;
    uint8_t*  d_outlier_sign = nullptr;

    UIntFlag  coeff_type     = UIntFlag::U64;
    UIntFlag  outlier_type   = UIntFlag::U64;
    PrecFlag  precision      = PrecFlag::F64;
    size_t    total_elems    = 0;   // padded coeff/sign count
    size_t    orig_elems     = 0;   // original (unpadded) element count
    size_t    num_words      = 0;   // ceil(total_elems / 64)
    uint64_t  num_outliers   = 0;
    double    q              = 0.0;
    double    mean           = 0.0;

    void free_device();
};

// ---------------------------------------------------------------------------
// CUDA kernels — templated on Real (float or double)
// All metadata (mean, q, tolerance) stays double for precision.
// Quantization arithmetic is always double regardless of Real.
// ---------------------------------------------------------------------------

template <typename Real>
__global__ void kernel_subtract_mean(Real* data, size_t n, double mean);

template <typename Real>
__global__ void kernel_subtract_mean_copy(Real* __restrict__ data,
                                          Real* __restrict__ orig,
                                          size_t n, double mean);

template <typename T, typename Real>
__global__ void kernel_midtread_quantize(const Real*   __restrict__ coeffs,
                                         T*            __restrict__ integers,
                                         uint64_t*     __restrict__ signs,
                                         size_t n, double inv_q);

template <typename T, typename Real>
__global__ void kernel_midtread_inv_quantize(const T*        __restrict__ integers,
                                             const uint64_t* __restrict__ signs,
                                             Real*           __restrict__ coeffs,
                                             size_t n, double q);

template <typename Real>
__global__ void kernel_outlier_flag_u8(const Real*   __restrict__ orig,
                                       const Real*   __restrict__ recon,
                                       uint8_t*      __restrict__ flags,
                                       size_t n, double tolerance);

template <typename T, typename Real>
__global__ void kernel_outlier_compact_u8(const Real*     __restrict__ orig,
                                          const Real*     __restrict__ recon,
                                          const uint8_t*  __restrict__ flags,
                                          const uint32_t* __restrict__ offsets,
                                          uint32_t*       __restrict__ out_idx,
                                          T*              __restrict__ out_val,
                                          uint8_t*        __restrict__ out_sign,
                                          size_t n, double inv_tol);

__global__ void kernel_outlier_count_from_last_u8(
    const uint32_t* __restrict__ offsets,
    const uint8_t*  __restrict__ flags,
    size_t n,
    uint64_t* __restrict__ out_count);

template <typename Real>
__global__ void kernel_add_mean(Real* data, size_t n, double mean);

template <typename T, typename Real>
__global__ void kernel_apply_outliers(Real*           __restrict__ data,
                                      const uint32_t* __restrict__ indices,
                                      const T*        __restrict__ values,
                                      const uint8_t*  __restrict__ signs,
                                      uint64_t n_outliers, double tolerance);

// Reductions: input is Real*, accumulation and output are always double
template <typename Real>
__global__ void kernel_abs_max(const Real* __restrict__ data, size_t n,
                               double* __restrict__ result);

template <typename Real>
__global__ void kernel_sum_reduce(const Real* __restrict__ data, size_t n,
                                  double* __restrict__ result);

// ---------------------------------------------------------------------------
// Shared utility
// ---------------------------------------------------------------------------
static inline size_t uint_elem_size(UIntFlag f)
{
    switch (f) {
        case UIntFlag::U8:  return 1;
        case UIntFlag::U16: return 2;
        case UIntFlag::U32: return 4;
        default:            return 8;
    }
}

// ---------------------------------------------------------------------------
// Reusable workspace — eliminates per-call cudaMalloc/cudaFree
// ---------------------------------------------------------------------------
struct QuantWorkspace {
    double*  d_reduce_buf[2] = {nullptr, nullptr};
    size_t   reduce_buf_elems = 0;

    void*    d_scan_temp      = nullptr;
    size_t   scan_temp_bytes  = 0;

    void alloc_reduce(size_t max_blocks);
    void ensure_scan_temp(size_t needed);
    void free_all();
};

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------
void save_coefficients(const std::string& path, const QuantResult& res);
void save_signs(const std::string& path, const QuantResult& res);
void save_combined(const std::string& path, const QuantResult& res);
void save_outliers(const std::string& path, const QuantResult& res);

#endif // WAVELET_QUANTIZER_CUH_
