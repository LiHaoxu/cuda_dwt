#ifndef PWE_KERNELS_CU_
#define PWE_KERNELS_CU_

#include "pwe_quantizer.cuh"
#include <cmath>
#include <cfloat>

// ============================================================================
// kernel_subtract_mean — subtract a scalar from every element
// ============================================================================
__global__ void kernel_subtract_mean(double* data, size_t n, double mean)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx < n)
        data[idx] -= mean;
}

// ============================================================================
// kernel_midtread_quantize — float → (unsigned integer, sign bitmask)
//
// Matches SPERR SPECK_FLT::m_midtread_quantize  (src/SPECK_FLT.cpp:310-370)
//   integer[i] = |round(coeff[i] / q)|
//   sign bit 1 = positive/zero, 0 = negative
//   Signs packed into uint64 words, LSB-first (SPERR Bitmask layout).
// ============================================================================
template <typename T>
__global__ void kernel_midtread_quantize(const double* __restrict__ coeffs,
                                         T*            __restrict__ integers,
                                         uint64_t*     __restrict__ signs,
                                         size_t n, double inv_q)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    double v   = coeffs[idx];
    long long ll = llrint(v * inv_q);
    T abs_val  = static_cast<T>(llabs(ll));
    integers[idx] = abs_val;

    // Pack sign bit into the correct uint64 word.
    // bit layout: word = idx/64, bit position = idx%64.  1 = non-negative.
    unsigned bit_pos = static_cast<unsigned>(idx & 63u);
    uint64_t mask    = uint64_t(ll >= 0) << bit_pos;

    // Use atomicOr so threads within the same warp/word don't race.
    atomicOr(reinterpret_cast<unsigned long long*>(&signs[idx >> 6]),
             static_cast<unsigned long long>(mask));
}

// ============================================================================
// kernel_midtread_inv_quantize — (unsigned integer, sign) → float
//
// Matches SPERR SPECK_FLT::m_midtread_inv_quantize (src/SPECK_FLT.cpp:372-398)
//   coeff[i] = q * integer[i] * (sign ? 1.0 : -1.0)
// ============================================================================
template <typename T>
__global__ void kernel_midtread_inv_quantize(const T*        __restrict__ integers,
                                             const uint64_t* __restrict__ signs,
                                             double*         __restrict__ coeffs,
                                             size_t n, double q)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    uint64_t word = signs[idx >> 6];
    unsigned bit  = static_cast<unsigned>(idx & 63u);
    double   sign = ((word >> bit) & 1ULL) ? 1.0 : -1.0;

    coeffs[idx] = q * static_cast<double>(integers[idx]) * sign;
}

// ============================================================================
// kernel_outlier_detect_quantize — compare orig vs recon, quantize outliers
//
// Matches SPERR PWE outlier logic (src/SPECK_FLT.cpp:460-484)
// and Outlier_Coder::m_quantize      (src/Outlier_Coder.cpp:188-204)
//
// For every element:
//   diff = orig[i] - recon[i]
//   if |diff| > tolerance:
//     out_ints[i]  = |round(diff / tolerance)|
//     out_signs[i] = (diff >= 0)
//     atomicAdd(out_count, 1)
//   else:
//     out_ints[i]  = 0
//     out_signs[i] = 1  (default positive, matching SPERR reset_true)
// ============================================================================
template <typename T>
__global__ void kernel_outlier_detect_quantize(const double*  __restrict__ orig,
                                               const double*  __restrict__ recon,
                                               T*             __restrict__ out_ints,
                                               uint64_t*      __restrict__ out_signs,
                                               unsigned long long* __restrict__ out_count,
                                               size_t n, double tolerance, double inv_tol)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    double diff = orig[idx] - recon[idx];
    unsigned bit_pos = static_cast<unsigned>(idx & 63u);

    if (fabs(diff) > tolerance) {
        long long ll = llrint(diff * inv_tol);
        out_ints[idx] = static_cast<T>(llabs(ll));

        // Sign: set bit if non-negative, clear if negative.
        // out_signs was pre-filled with all-1 bits, so we only need to clear
        // the bit for negative outliers.
        if (ll < 0) {
            atomicAnd(reinterpret_cast<unsigned long long*>(&out_signs[idx >> 6]),
                      ~static_cast<unsigned long long>(1ULL << bit_pos));
        }

        atomicAdd(out_count, 1ULL);
    } else {
        out_ints[idx] = T(0);
        // Sign bit stays 1 (pre-initialised).
    }
}

// ============================================================================
// kernel_abs_max — parallel reduction for max absolute value
//
// Two-phase: each block reduces to shared memory, then writes its partial
// result.  The host launches a second pass over the partial results.
// ============================================================================
__global__ void kernel_abs_max(const double* __restrict__ data, size_t n,
                               double* __restrict__ result)
{
    extern __shared__ double sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * size_t(blockDim.x * 2) + tid;

    double local_max = 0.0;
    if (idx < n) local_max = fabs(data[idx]);
    if (idx + blockDim.x < n) local_max = fmax(local_max, fabs(data[idx + blockDim.x]));

    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
template __global__ void kernel_midtread_quantize<uint8_t> (const double*, uint8_t*,  uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint16_t>(const double*, uint16_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint32_t>(const double*, uint32_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint64_t>(const double*, uint64_t*, uint64_t*, size_t, double);

template __global__ void kernel_midtread_inv_quantize<uint8_t> (const uint8_t*,  const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint16_t>(const uint16_t*, const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint32_t>(const uint32_t*, const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint64_t>(const uint64_t*, const uint64_t*, double*, size_t, double);

template __global__ void kernel_outlier_detect_quantize<uint8_t> (const double*, const double*, uint8_t*,  uint64_t*, unsigned long long*, size_t, double, double);
template __global__ void kernel_outlier_detect_quantize<uint16_t>(const double*, const double*, uint16_t*, uint64_t*, unsigned long long*, size_t, double, double);
template __global__ void kernel_outlier_detect_quantize<uint32_t>(const double*, const double*, uint32_t*, uint64_t*, unsigned long long*, size_t, double, double);
template __global__ void kernel_outlier_detect_quantize<uint64_t>(const double*, const double*, uint64_t*, uint64_t*, unsigned long long*, size_t, double, double);

#endif // PWE_KERNELS_CU_
