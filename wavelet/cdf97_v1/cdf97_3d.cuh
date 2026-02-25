#ifndef WAVELET_CDF97_3D_CUH_
#define WAVELET_CDF97_3D_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace wavelet {


static constexpr double ALPHA = -1.5861343420594238;
static constexpr double BETA  = -0.052980118573376651;
static constexpr double GAMMA =  0.88291107552850334;
static constexpr double DELTA =  0.44350685204498302;
static constexpr double K_SCALE     =  1.1496043988602962;
static constexpr double INV_K_SCALE =  1.0 / K_SCALE;


static constexpr int COMPUTE_X      = 32;
static constexpr int BANK_PADDING_X = 1;
static constexpr int TILE_X = COMPUTE_X + BANK_PADDING_X;  // 33
static constexpr int TILE_Y = 16;
static constexpr int VALID_PER_TILE = TILE_Y - 1;  // 15


static constexpr int BLOCK_SIZE = 256;

static_assert(TILE_X == 33);
static_assert(VALID_PER_TILE == 15);


enum class Axis : int { kX = 0, kY = 1, kZ = 2 };



//x + y*ldx + z*ldx*ldy.
__device__ __forceinline__
std::size_t Index3D(const int x, const int y, const int z,
                    const int ldx, const int ldy) {
    return static_cast<std::size_t>(x)
         + static_cast<std::size_t>(y) * static_cast<std::size_t>(ldx)
         + static_cast<std::size_t>(z) * static_cast<std::size_t>(ldx)
                                       * static_cast<std::size_t>(ldy);
}

__host__ __device__ __forceinline__
int ParityClamp(const int idx, const int n) {
    if (idx >= 0 && idx < n) return idx;
    const int p = idx & 1;            
    if (idx < 0) {
        return (p < n) ? p : 0;       // first element of same parity, else 0
    }
    // idx >= n
    int last = n - 1;
    if ((last & 1) != p) --last;      // step back to matching parity
    return (last >= 0) ? last : 0;
}


void dwt3d_forward(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

void dwt3d_inverse(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

}  

#endif  
