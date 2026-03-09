#include "cdf97_3d_f32.cuh"

namespace wavelet {

// =============================================================================
// X-direction FORWARD pass kernel (float, TILE_Y=32, BLOCK=512)
// =============================================================================
//
// Shared-memory tile:  s[32][33]  =  s[T_Y][T_X]
//   cols 0..31  = 32 compute points (16 even + 16 odd)
//   col  32     = bank-conflict padding
//   rows 0..31  = 32 Y-batch rows  (2x the double version)
//
// Grid:  (ceil(NX_L/32), ceil(NY_L/32), NZ_L)
// Block: (512, 1, 1)
// Thread mapping: row = tid >> 4 (0..31), k = tid & 0xF (0..15)

template <int COMP_X, int T_X, int T_Y, int BLK_SIZE>
__global__ void dwt3d_x_pass(float* __restrict__ data,
                              const int nx_l, const int ny_l, const int nz_l,
                              const int step,
                              const int ldx, const int ldy) {
    static constexpr int DATA_COLS = COMP_X;

    __shared__ float s[T_Y][T_X];

    const int tid = threadIdx.x;
    const int tile_start_x = static_cast<int>(blockIdx.x) * COMP_X;
    const int tile_start_y = static_cast<int>(blockIdx.y) * T_Y;
    const int iz           = static_cast<int>(blockIdx.z);

    const int n_valid_x = min(COMP_X, nx_l - tile_start_x);

    for (int idx = tid; idx < T_Y * DATA_COLS; idx += BLK_SIZE) {
        const int sy = idx / DATA_COLS;
        const int sx = idx - sy * DATA_COLS;

        const int lx = tile_start_x + sx;
        const int ly = tile_start_y + sy;

        s[sy][sx] = (lx < nx_l && ly < ny_l)
            ? data[Index3D(lx * step, ly * step, iz * step, ldx, ldy)]
            : 0.0f;
    }
    __syncthreads();

    const int row = tid >> 4;
    const int k   = tid & 0xF;

    const int pe = 2 * k;
    const int po = 2 * k + 1;

    if (po < n_valid_x) {
        float left  = s[row][po - 1];
        float right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] += fALPHA * (left + right);
    }
    __syncthreads();

    if (pe < n_valid_x) {
        float right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                            : (pe > 0 ? s[row][pe - 1] : 0.0f);
        float left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] += fBETA * (left + right);
    }
    __syncthreads();

    if (po < n_valid_x) {
        float left  = s[row][po - 1];
        float right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] += fGAMMA * (left + right);
    }
    __syncthreads();

    if (pe < n_valid_x) {
        float right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                            : (pe > 0 ? s[row][pe - 1] : 0.0f);
        float left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] = fK_SCALE * (s[row][pe] + fDELTA * (left + right));
    }
    __syncthreads();

    if (po < n_valid_x) {
        s[row][po] *= -fINV_K;
    }
    __syncthreads();

    const int ly = tile_start_y + row;
    if (ly < ny_l) {
        const int gy = ly * step;
        const int gz = iz * step;

        const int lx_even = tile_start_x + pe;
        if (lx_even < nx_l)
            data[Index3D(lx_even * step, gy, gz, ldx, ldy)] = s[row][pe];

        const int lx_odd = tile_start_x + po;
        if (lx_odd < nx_l)
            data[Index3D(lx_odd * step, gy, gz, ldx, ldy)] = s[row][po];
    }
}

// =============================================================================
// Y/Z-direction FORWARD pass kernel (float, TILE_Y=32, VPT=31, BLOCK=512)
// =============================================================================
//
// Thread mapping: x_base = tid & 0x1F (0..31), yz_idx = tid >> 5 (0..15)
//   Each thread covers 1 x position (stride=32 with X_BATCH=32).

template <Axis AXIS, int T_X, int T_Y, int VPT, int BLK_SIZE>
__global__ void dwt3d_yz_pass(float* __restrict__ data,
                               const int nx_l, const int ny_l, const int nz_l,
                               const int step,
                               const int ldx, const int ldy) {
    static constexpr int X_BATCH = T_X - 1;

    __shared__ float s[T_Y][T_X];

    const int tid          = threadIdx.x;
    const int tile_x_start = static_cast<int>(blockIdx.x) * X_BATCH;
    const int tile_c_idx   = static_cast<int>(blockIdx.y);
    const int batch_idx    = static_cast<int>(blockIdx.z);

    const int n_compute    = (AXIS == Axis::kY) ? ny_l : nz_l;
    const int tile_c_start = tile_c_idx * VPT;

    for (int idx = tid; idx < T_Y * T_X; idx += BLK_SIZE) {
        const int x_in_tile  = idx % T_X;
        const int yz_in_tile = idx / T_X;

        const int lx = tile_x_start + x_in_tile;
        const int lc = tile_c_start + yz_in_tile;

        if (x_in_tile < X_BATCH && yz_in_tile < VPT && lx < nx_l && lc < n_compute) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            s[yz_in_tile][x_in_tile] = data[Index3D(gx, gy, gz, ldx, ldy)];
        } else {
            s[yz_in_tile][x_in_tile] = 0.0f;
        }
    }
    __syncthreads();

    const int x_base = tid & 0x1F;
    const int yz_idx = tid >> 5;

    const int first_parity = tile_c_start & 1;
    const int first_odd    = (first_parity == 1) ? 0 : 1;
    const int first_even   = (first_parity == 0) ? 0 : 1;
    const int n_valid      = min(VPT, n_compute - tile_c_start);

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] += fALPHA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_even + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] += fBETA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] += fGAMMA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_even + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] = fK_SCALE * (s[tc][x] + fDELTA * (left + right));
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l)
                    s[tc][x] *= -fINV_K;
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < X_BATCH * n_valid; idx += BLK_SIZE) {
        const int sx       = idx % X_BATCH;
        const int yz_store = idx / X_BATCH;
        const int lx       = tile_x_start + sx;
        const int lc       = tile_c_start + yz_store;

        if (lx < nx_l) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            data[Index3D(gx, gy, gz, ldx, ldy)] = s[yz_store][sx];
        }
    }
}

// =============================================================================
// X-direction INVERSE pass kernel (float)
// =============================================================================

template <int COMP_X, int T_X, int T_Y, int BLK_SIZE>
__global__ void idwt3d_x_pass(float* __restrict__ data,
                               const int nx_l, const int ny_l, const int nz_l,
                               const int step,
                               const int ldx, const int ldy) {
    static constexpr int DATA_COLS = COMP_X;

    __shared__ float s[T_Y][T_X];

    const int tid = threadIdx.x;
    const int tile_start_x = static_cast<int>(blockIdx.x) * COMP_X;
    const int tile_start_y = static_cast<int>(blockIdx.y) * T_Y;
    const int iz           = static_cast<int>(blockIdx.z);

    const int n_valid_x = min(COMP_X, nx_l - tile_start_x);

    for (int idx = tid; idx < T_Y * DATA_COLS; idx += BLK_SIZE) {
        const int sy = idx / DATA_COLS;
        const int sx = idx - sy * DATA_COLS;

        const int lx = tile_start_x + sx;
        const int ly = tile_start_y + sy;

        s[sy][sx] = (lx < nx_l && ly < ny_l)
            ? data[Index3D(lx * step, ly * step, iz * step, ldx, ldy)]
            : 0.0f;
    }
    __syncthreads();

    const int row = tid >> 4;
    const int k   = tid & 0xF;
    const int pe = 2 * k;
    const int po = 2 * k + 1;

    if (po < n_valid_x)
        s[row][po] *= -fK_SCALE;
    __syncthreads();

    if (pe < n_valid_x) {
        float right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                            : (pe > 0 ? s[row][pe - 1] : 0.0f);
        float left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] = fINV_K * s[row][pe] - fDELTA * (left + right);
    }
    __syncthreads();

    if (po < n_valid_x) {
        float left  = s[row][po - 1];
        float right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] -= fGAMMA * (left + right);
    }
    __syncthreads();

    if (pe < n_valid_x) {
        float right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                            : (pe > 0 ? s[row][pe - 1] : 0.0f);
        float left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] -= fBETA * (left + right);
    }
    __syncthreads();

    if (po < n_valid_x) {
        float left  = s[row][po - 1];
        float right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] -= fALPHA * (left + right);
    }
    __syncthreads();

    const int ly = tile_start_y + row;
    if (ly < ny_l) {
        const int gy = ly * step;
        const int gz = iz * step;

        const int lx_even = tile_start_x + pe;
        if (lx_even < nx_l)
            data[Index3D(lx_even * step, gy, gz, ldx, ldy)] = s[row][pe];

        const int lx_odd = tile_start_x + po;
        if (lx_odd < nx_l)
            data[Index3D(lx_odd * step, gy, gz, ldx, ldy)] = s[row][po];
    }
}

// =============================================================================
// Y/Z-direction INVERSE pass kernel (float)
// =============================================================================

template <Axis AXIS, int T_X, int T_Y, int VPT, int BLK_SIZE>
__global__ void idwt3d_yz_pass(float* __restrict__ data,
                                const int nx_l, const int ny_l, const int nz_l,
                                const int step,
                                const int ldx, const int ldy) {
    static constexpr int X_BATCH = T_X - 1;

    __shared__ float s[T_Y][T_X];

    const int tid          = threadIdx.x;
    const int tile_x_start = static_cast<int>(blockIdx.x) * X_BATCH;
    const int tile_c_idx   = static_cast<int>(blockIdx.y);
    const int batch_idx    = static_cast<int>(blockIdx.z);

    const int n_compute    = (AXIS == Axis::kY) ? ny_l : nz_l;
    const int tile_c_start = tile_c_idx * VPT;

    for (int idx = tid; idx < T_Y * T_X; idx += BLK_SIZE) {
        const int x_in_tile  = idx % T_X;
        const int yz_in_tile = idx / T_X;

        const int lx = tile_x_start + x_in_tile;
        const int lc = tile_c_start + yz_in_tile;

        if (x_in_tile < X_BATCH && yz_in_tile < VPT && lx < nx_l && lc < n_compute) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            s[yz_in_tile][x_in_tile] = data[Index3D(gx, gy, gz, ldx, ldy)];
        } else {
            s[yz_in_tile][x_in_tile] = 0.0f;
        }
    }
    __syncthreads();

    const int x_base = tid & 0x1F;
    const int yz_idx = tid >> 5;

    const int first_parity = tile_c_start & 1;
    const int first_odd    = (first_parity == 1) ? 0 : 1;
    const int first_even   = (first_parity == 0) ? 0 : 1;
    const int n_valid      = min(VPT, n_compute - tile_c_start);

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l)
                    s[tc][x] *= -fK_SCALE;
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_even + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] = fINV_K * s[tc][x] - fDELTA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] -= fGAMMA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_even + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] -= fBETA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    {
        const int tc = first_odd + 2 * yz_idx;
        if (tc < n_valid) {
            for (int x = x_base; x < X_BATCH; x += T_Y) {
                if (tile_x_start + x < nx_l) {
                    float left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
                    float right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
                    s[tc][x] -= fALPHA * (left + right);
                }
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < X_BATCH * n_valid; idx += BLK_SIZE) {
        const int sx       = idx % X_BATCH;
        const int yz_store = idx / X_BATCH;
        const int lx       = tile_x_start + sx;
        const int lc       = tile_c_start + yz_store;

        if (lx < nx_l) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            data[Index3D(gx, gy, gz, ldx, ldy)] = s[yz_store][sx];
        }
    }
}

// =============================================================================
// 3-D symmetric reflection padding kernel (float)
// =============================================================================

__global__ void kernel_reflect_pad_3d(
    const float* __restrict__ src, float* __restrict__ dst,
    const int orig_nx, const int orig_ny, const int orig_nz,
    const int pad_nx,  const int pad_ny,  const int pad_nz)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t total = size_t(pad_nx) * pad_ny * pad_nz;
    if (idx >= total) return;

    int x = static_cast<int>(idx % pad_nx);
    int y = static_cast<int>((idx / pad_nx) % pad_ny);
    int z = static_cast<int>(idx / (size_t(pad_nx) * pad_ny));

    int sx = ReflectIdx(x, orig_nx);
    int sy = ReflectIdx(y, orig_ny);
    int sz = ReflectIdx(z, orig_nz);

    dst[idx] = src[size_t(sz) * orig_nx * orig_ny + size_t(sy) * orig_nx + sx];
}

// =============================================================================
// 3-D crop kernel (float)
// =============================================================================

__global__ void kernel_crop_3d(
    const float* __restrict__ src, float* __restrict__ dst,
    const int orig_nx, const int orig_ny, const int orig_nz,
    const int pad_nx,  const int pad_ny,  const int pad_nz)
{
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t total = size_t(orig_nx) * orig_ny * orig_nz;
    if (idx >= total) return;

    int x = static_cast<int>(idx % orig_nx);
    int y = static_cast<int>((idx / orig_nx) % orig_ny);
    int z = static_cast<int>(idx / (size_t(orig_nx) * orig_ny));

    dst[idx] = src[size_t(z) * pad_nx * pad_ny + size_t(y) * pad_nx + x];
}

}  // namespace wavelet
