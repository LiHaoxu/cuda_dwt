#ifndef WAVELET_CDF97_3D_CUH_
#define WAVELET_CDF97_3D_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace wavelet {

// =============================================================================
// CDF 9/7 Lifting Coefficients (analysis / forward)
// =============================================================================
static constexpr double ALPHA = -1.5861343420594238;
static constexpr double BETA  = -0.052980118573376651;
static constexpr double GAMMA =  0.88291107552850334;
static constexpr double DELTA =  0.44350685204498302;
static constexpr double K_SCALE     =  1.1496043988602962;
static constexpr double INV_K_SCALE =  1.0 / K_SCALE;

// =============================================================================
// Tile Constants — shared by X and Y/Z passes
// =============================================================================
//
// All passes share the same s[TILE_Y][TILE_X] = s[16][33] shared-memory tile.
// Each tile is self-contained: no halo from neighbouring tiles.
// Boundary extension uses tile-internal mirror clamp.
//
// X pass:
//   cols 0..31  = 32 compute points (16 even/odd pairs, pe=2k, po=2k+1)
//   col  32     = bank-conflict padding (never read during lifting)
//   rows 0..15  = 16 Y-direction batch rows
//
// Y/Z pass:
//   rows 0..14  = 15 valid output points (VALID_PER_TILE)
//   row  15     = padding (unused)
//   cols 0..31  = 32 X-direction batch positions
//   col  32     = bank-conflict padding

static constexpr int COMPUTE_X      = 32;
static constexpr int BANK_PADDING_X = 1;
static constexpr int TILE_X = COMPUTE_X + BANK_PADDING_X;  // 33
static constexpr int TILE_Y = 16;

// Y/Z valid output rows per tile (no halo, just 1 padding row)
static constexpr int VALID_PER_TILE = TILE_Y - 1;  // 15

// =============================================================================
// Common
// =============================================================================
static constexpr int BLOCK_SIZE = 256;

static_assert(TILE_X == 33);
static_assert(VALID_PER_TILE == 15);

// =============================================================================
// Axis Enum
// =============================================================================
enum class Axis : int { kX = 0, kY = 1, kZ = 2 };

// =============================================================================
// Symmetric reflection index for padding (shared by double and float kernels)
// =============================================================================
__device__ __forceinline__
int ReflectIdx(int i, int n) {
    if (n <= 1) return 0;
    int period = 2 * (n - 1);
    i = ((i % period) + period) % period;
    return (i < n) ? i : period - i;
}

// =============================================================================
// Device / Host Helper Functions
// =============================================================================


/// Row-major 3-D flat index:  x + y*ldx + z*ldx*ldy.
__device__ __forceinline__
std::size_t Index3D(const int x, const int y, const int z,
                    const int ldx, const int ldy) {
    return static_cast<std::size_t>(x)
         + static_cast<std::size_t>(y) * static_cast<std::size_t>(ldx)
         + static_cast<std::size_t>(z) * static_cast<std::size_t>(ldx)
                                       * static_cast<std::size_t>(ldy);
}

/// Clamp a logical index to [0, n-1], preserving even/odd parity.
/// This gives the correct CDF97 boundary treatment: an out-of-bounds even
/// sample is replaced by the last in-range even sample, and likewise for odd.
__host__ __device__ __forceinline__
int ParityClamp(const int idx, const int n) {
    if (idx >= 0 && idx < n) return idx;
    const int p = idx & 1;            // desired parity (works for negatives in 2's comp)
    if (idx < 0) {
        return (p < n) ? p : 0;       // first element of same parity, else 0
    }
    // idx >= n
    int last = n - 1;
    if ((last & 1) != p) --last;      // step back to matching parity
    return (last >= 0) ? last : 0;
}

// =============================================================================
// Public API
// =============================================================================

/// Forward 3-D multilevel CDF 9/7 DWT (analysis), in-place.
/// data  — device pointer, row-major layout with leading dims ldx, ldy.
/// nx, ny, nz — full physical sizes (must be even powers of 2).
/// levels     — decomposition depth.
void dwt3d_forward(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

/// Inverse 3-D multilevel CDF 9/7 DWT (synthesis), in-place.
/// Applies levels in reverse order; within each level: Z → Y → X.
void dwt3d_inverse(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

// =============================================================================
// Padding utilities for non-power-of-2 dimensions
// =============================================================================

/// Compute padded dimension: rounds up to next multiple of 2^levels.
int dwt3d_padded_dim(int dim, int levels);

/// 3D symmetric reflection padding: copy src (orig dims) into dst (pad dims).
/// src layout: orig_nx * orig_ny * orig_nz, row-major.
/// dst layout: pad_nx  * pad_ny  * pad_nz,  row-major.
void dwt3d_reflect_pad(const double* d_src, double* d_dst,
                       int orig_nx, int orig_ny, int orig_nz,
                       int pad_nx, int pad_ny, int pad_nz,
                       cudaStream_t stream = nullptr);

/// Crop padded data back to original dimensions.
void dwt3d_crop(const double* d_padded, double* d_dst,
                int orig_nx, int orig_ny, int orig_nz,
                int pad_nx, int pad_ny, int pad_nz,
                cudaStream_t stream = nullptr);

}  // namespace wavelet

#endif  // WAVELET_CDF97_3D_CUH_
