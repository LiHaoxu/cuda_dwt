#ifndef WAVELET_CDF97_3D_F32_CUH_
#define WAVELET_CDF97_3D_F32_CUH_

#include "cdf97_3d.cuh"

namespace wavelet {

// =============================================================================
// Float Tile Constants — same shared memory footprint as double, 2x rows
// =============================================================================
//
// double tile: s[16][33] * 8 = 4224 bytes
// float  tile: s[32][33] * 4 = 4224 bytes
//
// X pass:  32 compute cols × 32 Y-batch rows   (double: 16 rows)
// YZ pass: 31 valid output points × 32 X-batch  (double: 15 valid)

static constexpr int COMPUTE_X_F32      = 32;
static constexpr int BANK_PADDING_X_F32 = 1;
static constexpr int TILE_X_F32 = COMPUTE_X_F32 + BANK_PADDING_X_F32;  // 33
static constexpr int TILE_Y_F32 = 32;

static constexpr int VALID_PER_TILE_F32 = TILE_Y_F32 - 1;  // 31
static constexpr int BLOCK_SIZE_F32     = 512;

static_assert(sizeof(float) * TILE_Y_F32 * TILE_X_F32 ==
              sizeof(double) * TILE_Y * TILE_X,
              "float tile must use same shared memory as double tile");

// =============================================================================
// Float CDF 9/7 Coefficients (cast from double for compile-time precision)
// =============================================================================
static constexpr float fALPHA     = static_cast<float>(ALPHA);
static constexpr float fBETA      = static_cast<float>(BETA);
static constexpr float fGAMMA     = static_cast<float>(GAMMA);
static constexpr float fDELTA     = static_cast<float>(DELTA);
static constexpr float fK_SCALE   = static_cast<float>(K_SCALE);
static constexpr float fINV_K     = 1.0f / fK_SCALE;

// =============================================================================
// Public API — float overloads (same names as double versions)
// =============================================================================

void dwt3d_forward(float* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

void dwt3d_inverse(float* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

void dwt3d_reflect_pad(const float* d_src, float* d_dst,
                       int orig_nx, int orig_ny, int orig_nz,
                       int pad_nx, int pad_ny, int pad_nz,
                       cudaStream_t stream = nullptr);

void dwt3d_crop(const float* d_padded, float* d_dst,
                int orig_nx, int orig_ny, int orig_nz,
                int pad_nx, int pad_ny, int pad_nz,
                cudaStream_t stream = nullptr);

}  // namespace wavelet

#endif  // WAVELET_CDF97_3D_F32_CUH_
