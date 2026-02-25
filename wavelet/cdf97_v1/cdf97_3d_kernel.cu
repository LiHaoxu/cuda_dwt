
#include "cdf97_3d.cuh"

namespace wavelet {

//X
template <int COMP_X, int T_X, int T_Y, int BLK_SIZE>
__global__ void dwt3d_x_pass(double* __restrict__ data,
                              const int nx_l, const int ny_l, const int nz_l,
                              const int step,
                              const int ldx, const int ldy) {

    static constexpr int DATA_COLS = COMP_X;  // 32

    __shared__ double s[T_Y][T_X];

    const int tid = threadIdx.x;
    const int tile_start_x = static_cast<int>(blockIdx.x) * COMP_X;
    const int tile_start_y = static_cast<int>(blockIdx.y) * T_Y;
    const int iz           = static_cast<int>(blockIdx.z);

    // Number of valid data columns in this tile 
    const int n_valid_x = min(COMP_X, nx_l - tile_start_x);

    // LOAD: global → shared
    for (int idx = tid; idx < T_Y * DATA_COLS; idx += BLK_SIZE) {
        const int sy = idx / DATA_COLS;
        const int sx = idx - sy * DATA_COLS;          // idx % DATA_COLS

        const int lx = tile_start_x + sx;
        const int ly = tile_start_y + sy;

        s[sy][sx] = (lx < nx_l && ly < ny_l)
            ? data[Index3D(lx * step, ly * step, iz * step, ldx, ldy)]
            : 0.0;
    }
    __syncthreads();



    const int row = tid >> 4;           // 0..15
    const int k   = tid & 0xF;         // 0..15

    const int pe = 2 * k;              // even position in shared
    const int po = 2 * k + 1;          // odd  position in shared

    // Step 1  predict:  odd[k]  += α · (even[k] + even[k+1])
    if (po < n_valid_x) {
        double left  = s[row][po - 1];
        double right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] += ALPHA * (left + right);
    }
    __syncthreads();

    // Step 2  update:   even[k] += β · (odd[k−1] + odd[k])
    if (pe < n_valid_x) {
        double right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                             : (pe > 0 ? s[row][pe - 1] : 0.0);
        double left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] += BETA * (left + right);
    }
    __syncthreads();

    // Step 3  predict:  odd[k]  += γ · (even[k] + even[k+1])
    if (po < n_valid_x) {
        double left  = s[row][po - 1];
        double right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] += GAMMA * (left + right);
    }
    __syncthreads();

    // Step 4  update + scale:  even[k] = K · (even[k] + δ · (odd[k−1]+odd[k]))
    if (pe < n_valid_x) {
        double right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                             : (pe > 0 ? s[row][pe - 1] : 0.0);
        double left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] = K_SCALE * (s[row][pe] + DELTA * (left + right));
    }
    __syncthreads();

    // Step 5  scale odd:  odd[k] *= −1/K
    if (po < n_valid_x) {
        s[row][po] *= -INV_K_SCALE;
    }
    __syncthreads();

    // STORE: shared → global 
    const int ly = tile_start_y + row;
    if (ly < ny_l) {
        const int gy = ly * step;
        const int gz = iz * step;

        const int lx_even = tile_start_x + pe;
        if (lx_even < nx_l) {
            data[Index3D(lx_even * step, gy, gz, ldx, ldy)] = s[row][pe];
        }
        const int lx_odd = tile_start_x + po;
        if (lx_odd < nx_l) {
            data[Index3D(lx_odd * step, gy, gz, ldx, ldy)] = s[row][po];
        }
    }
}


//Y/Z
template <Axis AXIS, int T_X, int T_Y, int VPT, int BLK_SIZE>
__global__ void dwt3d_yz_pass(double* __restrict__ data,
                               const int nx_l, const int ny_l, const int nz_l,
                               const int step,
                               const int ldx, const int ldy) {
    // X batch size = T_X − 1 
    static constexpr int X_BATCH = T_X - 1;  // 32

    __shared__ double s[T_Y][T_X];

    const int tid          = threadIdx.x;
    const int tile_x_start = static_cast<int>(blockIdx.x) * X_BATCH;
    const int tile_c_idx   = static_cast<int>(blockIdx.y);
    const int batch_idx    = static_cast<int>(blockIdx.z);

    const int n_compute    = (AXIS == Axis::kY) ? ny_l : nz_l;
    const int tile_c_start = tile_c_idx * VPT;  // first output's logical index

    //  LOAD: global → shared 

    for (int idx = tid; idx < T_Y * T_X; idx += BLK_SIZE) {
        const int x_in_tile  = idx % T_X;          // 0..32
        const int yz_in_tile = idx / T_X;          // 0..15

        const int lx = tile_x_start + x_in_tile;
        const int lc = tile_c_start + yz_in_tile;  // no halo offset

        if (yz_in_tile < VPT && lx < nx_l && lc < n_compute) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            s[yz_in_tile][x_in_tile] = data[Index3D(gx, gy, gz, ldx, ldy)];
        } else {
            s[yz_in_tile][x_in_tile] = 0.0;
        }
    }
    __syncthreads();

//compute
    const int x      = tid % T_X;          // 0..32
    const int yz_idx = tid / T_X;          // 0..7  (256/33 ≈ 7.75)

    const bool x_active = (x < X_BATCH) && (tile_x_start + x < nx_l);

    const int first_parity = tile_c_start & 1;            // row 0's global parity
    const int first_odd    = (first_parity == 1) ? 0 : 1; // first globally-odd row
    const int first_even   = (first_parity == 0) ? 0 : 1; // first globally-even row
    const int n_valid      = min(VPT, n_compute - tile_c_start);

    // Step 1  predict 
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] += ALPHA * (left + right);
        }
    }
    __syncthreads();

    // Step 2  update
    {
        const int tc = first_even + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] += BETA * (left + right);
        }
    }
    __syncthreads();

    // Step 3  predict
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] += GAMMA * (left + right);
        }
    }
    __syncthreads();

    // Step 4  update + scale
    {
        const int tc = first_even + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] = K_SCALE * (s[tc][x] + DELTA * (left + right));
        }
    }
    __syncthreads();

    // Step 5  scale odd
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid)
            s[tc][x] *= -INV_K_SCALE;
    }
    __syncthreads();

    //STORE: shared → global 
    for (int idx = tid; idx < X_BATCH * n_valid; idx += BLK_SIZE) {
        const int sx       = idx % X_BATCH;          // 0..31
        const int yz_store = idx / X_BATCH;          // 0..n_valid-1
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

//X inverse

template <int COMP_X, int T_X, int T_Y, int BLK_SIZE>
__global__ void idwt3d_x_pass(double* __restrict__ data,
                               const int nx_l, const int ny_l, const int nz_l,
                               const int step,
                               const int ldx, const int ldy) {
    static constexpr int DATA_COLS = COMP_X;  // 32

    __shared__ double s[T_Y][T_X];

    const int tid = threadIdx.x;
    const int tile_start_x = static_cast<int>(blockIdx.x) * COMP_X;
    const int tile_start_y = static_cast<int>(blockIdx.y) * T_Y;
    const int iz           = static_cast<int>(blockIdx.z);

    // Number of valid data columns in this tile.
    const int n_valid_x = min(COMP_X, nx_l - tile_start_x);

    for (int idx = tid; idx < T_Y * DATA_COLS; idx += BLK_SIZE) {
        const int sy = idx / DATA_COLS;
        const int sx = idx - sy * DATA_COLS;

        const int lx = tile_start_x + sx;
        const int ly = tile_start_y + sy;

        s[sy][sx] = (lx < nx_l && ly < ny_l)
            ? data[Index3D(lx * step, ly * step, iz * step, ldx, ldy)]
            : 0.0;
    }
    __syncthreads();

//compute
    const int row = tid >> 4;
    const int k   = tid & 0xF;

    const int pe = 2 * k;
    const int po = 2 * k + 1;

    // Inv-step 1  (undo fwd-step 5):  odd[k] *= −K
    if (po < n_valid_x) {
        s[row][po] *= -K_SCALE;
    }
    __syncthreads();

    // Inv-step 2  (undo fwd-step 4):  even[k] = even[k]/K − δ·(odd[k−1]+odd[k])
    if (pe < n_valid_x) {
        double right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                             : (pe > 0 ? s[row][pe - 1] : 0.0);
        double left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] = INV_K_SCALE * s[row][pe] - DELTA * (left + right);
    }
    __syncthreads();

    // Inv-step 3  (undo fwd-step 3):  odd[k] −= γ·(even[k]+even[k+1])
    if (po < n_valid_x) {
        double left  = s[row][po - 1];
        double right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] -= GAMMA * (left + right);
    }
    __syncthreads();

    // Inv-step 4  (undo fwd-step 2):  even[k] −= β·(odd[k−1]+odd[k])
    if (pe < n_valid_x) {
        double right = (pe + 1 < n_valid_x) ? s[row][pe + 1]
                                             : (pe > 0 ? s[row][pe - 1] : 0.0);
        double left  = (pe > 0) ? s[row][pe - 1] : right;
        s[row][pe] -= BETA * (left + right);
    }
    __syncthreads();

    // Inv-step 5  (undo fwd-step 1):  odd[k] −= α·(even[k]+even[k+1])
    if (po < n_valid_x) {
        double left  = s[row][po - 1];
        double right = (po < n_valid_x - 1) ? s[row][po + 1] : left;
        s[row][po] -= ALPHA * (left + right);
    }
    __syncthreads();

    const int ly = tile_start_y + row;
    if (ly < ny_l) {
        const int gy = ly * step;
        const int gz = iz * step;

        const int lx_even = tile_start_x + pe;
        if (lx_even < nx_l) {
            data[Index3D(lx_even * step, gy, gz, ldx, ldy)] = s[row][pe];
        }
        const int lx_odd = tile_start_x + po;
        if (lx_odd < nx_l) {
            data[Index3D(lx_odd * step, gy, gz, ldx, ldy)] = s[row][po];
        }
    }
}

//YZ inverse

template <Axis AXIS, int T_X, int T_Y, int VPT, int BLK_SIZE>
__global__ void idwt3d_yz_pass(double* __restrict__ data,
                                const int nx_l, const int ny_l, const int nz_l,
                                const int step,
                                const int ldx, const int ldy) {
    static constexpr int X_BATCH = T_X - 1;  // 32

    __shared__ double s[T_Y][T_X];

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

        if (yz_in_tile < VPT && lx < nx_l && lc < n_compute) {
            int gx, gy, gz;
            if constexpr (AXIS == Axis::kY) {
                gx = lx * step;  gy = lc * step;  gz = batch_idx * step;
            } else {
                gx = lx * step;  gy = batch_idx * step;  gz = lc * step;
            }
            s[yz_in_tile][x_in_tile] = data[Index3D(gx, gy, gz, ldx, ldy)];
        } else {
            s[yz_in_tile][x_in_tile] = 0.0;
        }
    }
    __syncthreads();

//compute
    const int x      = tid % T_X;
    const int yz_idx = tid / T_X;

    const bool x_active = (x < X_BATCH) && (tile_x_start + x < nx_l);

    const int first_parity = tile_c_start & 1;
    const int first_odd    = (first_parity == 1) ? 0 : 1;
    const int first_even   = (first_parity == 0) ? 0 : 1;
    const int n_valid      = min(VPT, n_compute - tile_c_start);

    // Inv-step 1  (undo fwd-step 5):  odd *= −K
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid)
            s[tc][x] *= -K_SCALE;
    }
    __syncthreads();

    // Inv-step 2  (undo fwd-step 4):  even = even/K − δ·(odd_left+odd_right)
    {
        const int tc = first_even + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] = INV_K_SCALE * s[tc][x] - DELTA * (left + right);
        }
    }
    __syncthreads();

    // Inv-step 3  (undo fwd-step 3):  odd −= γ·(even_left+even_right)
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] -= GAMMA * (left + right);
        }
    }
    __syncthreads();

    // Inv-step 4  (undo fwd-step 2):  even −= β·(odd_left+odd_right)
    {
        const int tc = first_even + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] -= BETA * (left + right);
        }
    }
    __syncthreads();

    // Inv-step 5  (undo fwd-step 1):  odd −= α·(even_left+even_right)
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] -= ALPHA * (left + right);
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

} 
