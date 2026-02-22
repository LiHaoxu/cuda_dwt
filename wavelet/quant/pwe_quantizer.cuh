#ifndef PWE_QUANTIZER_CUH_
#define PWE_QUANTIZER_CUH_

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

struct PweConfig {
    int    nx       = 0;
    int    ny       = 0;
    int    nz       = 0;
    int    levels   = 0;
    double tolerance = 0.0;
};

struct PweResult {
    void*     d_coeffs      = nullptr;  // device: quantized integers (uint8/16/32/64)
    uint64_t* d_signs        = nullptr;  // device: sign bitmask (packed uint64 words)
    void*     d_out_coeffs   = nullptr;  // device: outlier integers (sparse, same type)
    uint64_t* d_out_signs    = nullptr;  // device: outlier sign bitmask
    UIntFlag  coeff_type     = UIntFlag::U64;
    UIntFlag  outlier_type   = UIntFlag::U64;
    size_t    total_elems    = 0;
    size_t    num_words      = 0;        // ceil(total_elems / 64)
    uint64_t  num_outliers   = 0;
    double    q              = 0.0;
    double    mean           = 0.0;

    void free_device();
};

// ---------------------------------------------------------------------------
// CUDA kernels (defined in pwe_kernels.cu)
// ---------------------------------------------------------------------------

__global__ void kernel_subtract_mean(double* data, size_t n, double mean);

template <typename T>
__global__ void kernel_midtread_quantize(const double* __restrict__ coeffs,
                                         T*            __restrict__ integers,
                                         uint64_t*     __restrict__ signs,
                                         size_t n, double inv_q);

template <typename T>
__global__ void kernel_midtread_inv_quantize(const T*        __restrict__ integers,
                                             const uint64_t* __restrict__ signs,
                                             double*         __restrict__ coeffs,
                                             size_t n, double q);

template <typename T>
__global__ void kernel_outlier_detect_quantize(const double*  __restrict__ orig,
                                               const double*  __restrict__ recon,
                                               T*             __restrict__ out_ints,
                                               uint64_t*      __restrict__ out_signs,
                                               unsigned long long* __restrict__ out_count,
                                               size_t n, double tolerance, double inv_tol);

__global__ void kernel_abs_max(const double* __restrict__ data, size_t n,
                               double* __restrict__ result);

// ---------------------------------------------------------------------------
// Host helpers (defined in pwe_quantizer.cu)
// ---------------------------------------------------------------------------

void save_coefficients(const std::string& path, const PweResult& res);
void save_signs(const std::string& path, const PweResult& res);
void save_outliers(const std::string& path, const PweResult& res);

#endif // PWE_QUANTIZER_CUH_
