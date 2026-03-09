#ifndef QUANT_KERNELS_CU_
#define QUANT_KERNELS_CU_

#include "quantizer.cuh"  // found via -I include/
#include <cmath>
#include <cfloat>

// ============================================================================
// Conditioning kernels
// ============================================================================

template <typename Real>
__global__ void kernel_subtract_mean(Real* data, size_t n, double mean)
{
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         idx < n;
         idx += size_t(gridDim.x) * blockDim.x) {
        data[idx] = static_cast<Real>(static_cast<double>(data[idx]) - mean);
    }
}

template <typename Real>
__global__ void kernel_subtract_mean_copy(Real* __restrict__ data,
                                          Real* __restrict__ orig,
                                          size_t n, double mean)
{
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         idx < n;
         idx += size_t(gridDim.x) * blockDim.x) {
        Real v = static_cast<Real>(static_cast<double>(data[idx]) - mean);
        data[idx] = v;
        orig[idx] = v;
    }
}

// ============================================================================
// Midtread quantization kernels
//
// Quantization arithmetic always in double regardless of Real.
// ============================================================================

template <typename T, typename Real>
__global__ void kernel_midtread_quantize(const Real*   __restrict__ coeffs,
                                         T*            __restrict__ integers,
                                         uint64_t*     __restrict__ signs,
                                         size_t n, double inv_q)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

    long long ll = 0;
    if (idx < n) {
        double v = static_cast<double>(coeffs[idx]);
        ll = __double2ll_rn(v * inv_q);
        integers[idx] = static_cast<T>(llabs(ll));
    }

    unsigned ballot = __ballot_sync(0xFFFFFFFF, (idx < n) && (ll >= 0));

    unsigned lane = threadIdx.x & 31;
    if (lane == 0) {
        size_t half_word     = idx >> 5;
        size_t num_half_words = ((n + 63) >> 6) << 1;
        if (half_word < num_half_words)
            reinterpret_cast<uint32_t*>(signs)[half_word] = ballot;
    }
}

template <typename T, typename Real>
__global__ void kernel_midtread_inv_quantize(const T*        __restrict__ integers,
                                             const uint64_t* __restrict__ signs,
                                             Real*           __restrict__ coeffs,
                                             size_t n, double q)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    uint64_t word = signs[idx >> 6];
    unsigned bit  = static_cast<unsigned>(idx & 63u);
    double   sign = ((word >> bit) & 1ULL) ? 1.0 : -1.0;

    coeffs[idx] = static_cast<Real>(q * static_cast<double>(integers[idx]) * sign);
}

// ============================================================================
// Outlier detection & compaction
// ============================================================================

template <typename Real>
__global__ void kernel_outlier_flag_u8(const Real*   __restrict__ orig,
                                       const Real*   __restrict__ recon,
                                       uint8_t*      __restrict__ flags,
                                       size_t n, double tolerance)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx < n) {
        double diff = static_cast<double>(orig[idx]) - static_cast<double>(recon[idx]);
        flags[idx] = (fabs(diff) > tolerance) ? uint8_t(1) : uint8_t(0);
    }
}

template <typename T, typename Real>
__global__ void kernel_outlier_compact_u8(const Real*     __restrict__ orig,
                                          const Real*     __restrict__ recon,
                                          const uint8_t*  __restrict__ flags,
                                          const uint32_t* __restrict__ offsets,
                                          uint32_t*       __restrict__ out_idx,
                                          T*              __restrict__ out_val,
                                          uint8_t*        __restrict__ out_sign,
                                          size_t n, double inv_tol)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n || flags[idx] == 0) return;

    uint32_t pos  = offsets[idx];
    double   diff = static_cast<double>(orig[idx]) - static_cast<double>(recon[idx]);
    long long ll  = __double2ll_rn(diff * inv_tol);

    out_idx[pos]  = static_cast<uint32_t>(idx);
    out_val[pos]  = static_cast<T>(llabs(ll));
    out_sign[pos] = (ll >= 0) ? uint8_t(1) : uint8_t(0);
}

__global__ void kernel_outlier_count_from_last_u8(
    const uint32_t* __restrict__ offsets,
    const uint8_t*  __restrict__ flags,
    size_t n,
    uint64_t* __restrict__ out_count)
{
    *out_count = uint64_t(offsets[n - 1]) + uint64_t(flags[n - 1]);
}

// ============================================================================
// Decoding kernels
// ============================================================================

template <typename Real>
__global__ void kernel_add_mean(Real* data, size_t n, double mean)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx < n)
        data[idx] = static_cast<Real>(static_cast<double>(data[idx]) + mean);
}

template <typename T, typename Real>
__global__ void kernel_apply_outliers(Real*           __restrict__ data,
                                      const uint32_t* __restrict__ indices,
                                      const T*        __restrict__ values,
                                      const uint8_t*  __restrict__ signs,
                                      uint64_t n_outliers, double tolerance)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n_outliers) return;

    uint32_t pos = indices[idx];
    double val   = static_cast<double>(values[idx]) * tolerance;
    double sign  = signs[idx] ? 1.0 : -1.0;
    data[pos] = static_cast<Real>(static_cast<double>(data[pos]) + val * sign);
}

// ============================================================================
// Parallel reductions — input Real*, accumulation and output always double
// ============================================================================

template <typename Real>
__global__ void kernel_abs_max(const Real* __restrict__ data, size_t n,
                               double* __restrict__ result)
{
    extern __shared__ double sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * size_t(blockDim.x * 2) + tid;

    double local_max = 0.0;
    if (idx < n) local_max = fabs(static_cast<double>(data[idx]));
    if (idx + blockDim.x < n) local_max = fmax(local_max, fabs(static_cast<double>(data[idx + blockDim.x])));

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

template <typename Real>
__global__ void kernel_sum_reduce(const Real* __restrict__ data, size_t n,
                                  double* __restrict__ result)
{
    extern __shared__ double sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * size_t(blockDim.x * 2) + tid;

    double local_sum = 0.0;
    if (idx < n) local_sum = static_cast<double>(data[idx]);
    if (idx + blockDim.x < n) local_sum += static_cast<double>(data[idx + blockDim.x]);

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

// ============================================================================
// Explicit template instantiations — {float, double} x {UInt types}
// ============================================================================

// --- subtract_mean ---
template __global__ void kernel_subtract_mean<float> (float*,  size_t, double);
template __global__ void kernel_subtract_mean<double>(double*, size_t, double);

// --- subtract_mean_copy ---
template __global__ void kernel_subtract_mean_copy<float> (float*,  float*,  size_t, double);
template __global__ void kernel_subtract_mean_copy<double>(double*, double*, size_t, double);

// --- midtread_quantize (UInt x Real) ---
template __global__ void kernel_midtread_quantize<uint8_t,  float> (const float*,  uint8_t*,  uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint16_t, float> (const float*,  uint16_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint32_t, float> (const float*,  uint32_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint64_t, float> (const float*,  uint64_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint8_t,  double>(const double*, uint8_t*,  uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint16_t, double>(const double*, uint16_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint32_t, double>(const double*, uint32_t*, uint64_t*, size_t, double);
template __global__ void kernel_midtread_quantize<uint64_t, double>(const double*, uint64_t*, uint64_t*, size_t, double);

// --- midtread_inv_quantize (UInt x Real) ---
template __global__ void kernel_midtread_inv_quantize<uint8_t,  float> (const uint8_t*,  const uint64_t*, float*,  size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint16_t, float> (const uint16_t*, const uint64_t*, float*,  size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint32_t, float> (const uint32_t*, const uint64_t*, float*,  size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint64_t, float> (const uint64_t*, const uint64_t*, float*,  size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint8_t,  double>(const uint8_t*,  const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint16_t, double>(const uint16_t*, const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint32_t, double>(const uint32_t*, const uint64_t*, double*, size_t, double);
template __global__ void kernel_midtread_inv_quantize<uint64_t, double>(const uint64_t*, const uint64_t*, double*, size_t, double);

// --- outlier_flag_u8 ---
template __global__ void kernel_outlier_flag_u8<float> (const float*,  const float*,  uint8_t*, size_t, double);
template __global__ void kernel_outlier_flag_u8<double>(const double*, const double*, uint8_t*, size_t, double);

// --- outlier_compact_u8 (UInt x Real) ---
template __global__ void kernel_outlier_compact_u8<uint8_t,  float> (const float*,  const float*,  const uint8_t*, const uint32_t*, uint32_t*, uint8_t*,  uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint16_t, float> (const float*,  const float*,  const uint8_t*, const uint32_t*, uint32_t*, uint16_t*, uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint32_t, float> (const float*,  const float*,  const uint8_t*, const uint32_t*, uint32_t*, uint32_t*, uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint64_t, float> (const float*,  const float*,  const uint8_t*, const uint32_t*, uint32_t*, uint64_t*, uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint8_t,  double>(const double*, const double*, const uint8_t*, const uint32_t*, uint32_t*, uint8_t*,  uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint16_t, double>(const double*, const double*, const uint8_t*, const uint32_t*, uint32_t*, uint16_t*, uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint32_t, double>(const double*, const double*, const uint8_t*, const uint32_t*, uint32_t*, uint32_t*, uint8_t*, size_t, double);
template __global__ void kernel_outlier_compact_u8<uint64_t, double>(const double*, const double*, const uint8_t*, const uint32_t*, uint32_t*, uint64_t*, uint8_t*, size_t, double);

// --- add_mean ---
template __global__ void kernel_add_mean<float> (float*,  size_t, double);
template __global__ void kernel_add_mean<double>(double*, size_t, double);

// --- apply_outliers (UInt x Real) ---
template __global__ void kernel_apply_outliers<uint8_t,  float> (float*,  const uint32_t*, const uint8_t*,  const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint16_t, float> (float*,  const uint32_t*, const uint16_t*, const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint32_t, float> (float*,  const uint32_t*, const uint32_t*, const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint64_t, float> (float*,  const uint32_t*, const uint64_t*, const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint8_t,  double>(double*, const uint32_t*, const uint8_t*,  const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint16_t, double>(double*, const uint32_t*, const uint16_t*, const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint32_t, double>(double*, const uint32_t*, const uint32_t*, const uint8_t*, uint64_t, double);
template __global__ void kernel_apply_outliers<uint64_t, double>(double*, const uint32_t*, const uint64_t*, const uint8_t*, uint64_t, double);

// --- abs_max ---
template __global__ void kernel_abs_max<float> (const float*,  size_t, double*);
template __global__ void kernel_abs_max<double>(const double*, size_t, double*);

// --- sum_reduce ---
template __global__ void kernel_sum_reduce<float> (const float*,  size_t, double*);
template __global__ void kernel_sum_reduce<double>(const double*, size_t, double*);

#endif // QUANT_KERNELS_CU_
