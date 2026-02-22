// wavelet/cdf97_3d_kernels.cu  —  X / Y / Z pass kernels for 3-D CDF 9/7 DWT
//
// This file is meant to be #included from cdf97_3d_host.cu so that the
// template kernel bodies are visible at the launch site.
//
// v2: Halo-free, tile-self-contained design.
//     Each tile does NOT reference neighbouring tiles.
//     Boundary extension uses tile-internal mirror clamp.
//     Forward and inverse use identical clamp rules → perfect roundtrip.

#include "cdf97_3d.cuh"

namespace wavelet {

// =============================================================================
// X-direction FORWARD pass kernel  (analysis)
// =============================================================================
//
// Shared-memory tile:  s[T_Y][T_X]  =  s[16][33]
//   cols 0..31   — 32 compute points (16 even + 16 odd, pe=2k, po=2k+1)
//   col  32      — bank-conflict padding (never read during lifting)
//
// Grid:  (ceil(NX_L/32), ceil(NY_L/16), NZ_L)
// Block: (256, 1, 1)
// Thread mapping for compute:  row = tid/16,  k = tid%16  → 256 = 16×16
//
// Boundary clamp (tile-internal, no halo):
//   predict:  left = even[k] (always valid),
//             right = even[k+1] (k<15) or even[k] (k=15)
//   update:   right = odd[k] (always valid),
//             left = odd[k-1] (k>0) or odd[k] (k=0)

template <int COMP_X, int T_X, int T_Y, int BLK_SIZE>
__global__ void dwt3d_x_pass(double* __restrict__ data,
                              const int nx_l, const int ny_l, const int nz_l,
                              const int step,
                              const int ldx, const int ldy) {
    // 32 data columns, no halo.
    static constexpr int DATA_COLS = COMP_X;  // 32

    __shared__ double s[T_Y][T_X];

    const int tid = threadIdx.x;
    const int tile_start_x = static_cast<int>(blockIdx.x) * COMP_X;
    const int tile_start_y = static_cast<int>(blockIdx.y) * T_Y;
    const int iz           = static_cast<int>(blockIdx.z);

    // Number of valid data columns in this tile (may be < 32 for the last tile).
    const int n_valid_x = min(COMP_X, nx_l - tile_start_x);

    // ---- LOAD: global → shared (32 cols × 16 rows = 512 elements) ----
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

    // ---- COMPUTE: 5-step CDF 9/7 lifting (in shared memory) ----
    //
    // even[k] ≡ s[row][2k]      (k = 0..15,  shared cols 0,2,4,…,30)
    // odd[k]  ≡ s[row][2k+1]    (k = 0..15,  shared cols 1,3,5,…,31)
    //
    // n_valid_x guards: only positions < n_valid_x participate in lifting.
    // Boundary clamp (within valid range):
    //   predict:  left = even[k] (always valid when po < n_valid_x),
    //             right = even[k+1] if po+1 < n_valid_x, else mirror left
    //   update:   right = odd[k] if pe+1 < n_valid_x, else mirror left,
    //             left = odd[k-1] if pe > 0, else mirror right

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

    // ---- STORE: shared → global (in-place even/odd) ----
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

// =============================================================================
// Y/Z-direction FORWARD pass kernel  (analysis, template on Axis)
// =============================================================================
//
// Reuses the SAME 33×16 tile as the X pass, with swapped semantics:
//
//   Shared memory:  s[T_Y][T_X]  =  s[16][33]
//     row (T_Y=16) = Y/Z compute direction
//       rows 0..14 — 15 valid output points (VPT)
//       row  15    — padding (unused)
//     col (T_X=33) = X batch direction
//       cols 0..31 — 32 X batch positions
//       col  32    — bank-conflict padding
//
// No halo: boundary extension via tile-internal mirror clamp.
//
// Grid (Y pass):  (ceil(NX_L/32), ceil(NY_L/15), NZ_L)
// Grid (Z pass):  (ceil(NX_L/32), ceil(NZ_L/15), NY_L)
// Block: (256, 1, 1)
//
// Thread mapping:
//   Load / Store:  loop over elements
//   Compute:  x = tid % T_X (0..32),  yz_idx = tid / T_X (0..7)

template <Axis AXIS, int T_X, int T_Y, int VPT, int BLK_SIZE>
__global__ void dwt3d_yz_pass(double* __restrict__ data,
                               const int nx_l, const int ny_l, const int nz_l,
                               const int step,
                               const int ldx, const int ldy) {
    // X batch size = T_X − 1 (last column is bank padding).
    static constexpr int X_BATCH = T_X - 1;  // 32

    __shared__ double s[T_Y][T_X];

    const int tid          = threadIdx.x;
    const int tile_x_start = static_cast<int>(blockIdx.x) * X_BATCH;
    const int tile_c_idx   = static_cast<int>(blockIdx.y);
    const int batch_idx    = static_cast<int>(blockIdx.z);

    const int n_compute    = (AXIS == Axis::kY) ? ny_l : nz_l;
    const int tile_c_start = tile_c_idx * VPT;  // first output's logical index

    // ---- LOAD: global → shared (T_Y × T_X = 528 elements) ----
    // No halo: lc = tile_c_start + yz_in_tile  (no −1 offset).
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

    // ---- COMPUTE: 5-step CDF 9/7 lifting along Y/Z (row direction) ----
    //
    // Thread mapping: x = tid % T_X (0..32), yz_idx = tid / T_X (0..7).
    //
    // Parity: global_index(row r) = tile_c_start + r.
    //   If tile_c_start is even: row 0 is globally even, row 1 is globally odd, ...
    //   If tile_c_start is odd:  row 0 is globally odd,  row 1 is globally even, ...
    //
    // Boundary clamp (tile-internal mirror):
    //   left  = (tc > 0)          ? s[tc-1][x] : s[tc+1][x]
    //   right = (tc < n_valid-1)  ? s[tc+1][x] : left

    const int x      = tid % T_X;          // 0..32
    const int yz_idx = tid / T_X;          // 0..7  (256/33 ≈ 7.75)

    const bool x_active = (x < X_BATCH) && (tile_x_start + x < nx_l);

    const int first_parity = tile_c_start & 1;            // row 0's global parity
    const int first_odd    = (first_parity == 1) ? 0 : 1; // first globally-odd row
    const int first_even   = (first_parity == 0) ? 0 : 1; // first globally-even row
    const int n_valid      = min(VPT, n_compute - tile_c_start);

    // Step 1  predict (globally-odd)
    {
        const int tc = first_odd + 2 * yz_idx;
        if (x_active && tc < n_valid) {
            double left  = (tc > 0)          ? s[tc - 1][x] : s[tc + 1][x];
            double right = (tc < n_valid - 1) ? s[tc + 1][x] : left;
            s[tc][x] += ALPHA * (left + right);
        }
    }
    __syncthreads();

    // Step 2  update (globally-even)
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

    // ---- STORE: shared → global (all valid output rows) ----
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

// =============================================================================
// X-direction INVERSE pass kernel  (synthesis / reconstruction)
// =============================================================================
//
// Identical tile layout and load/store as dwt3d_x_pass.
// Same boundary clamp rules.  Only the 5 lifting steps are reversed:
//   fwd:  predict(α) → update(β) → predict(γ) → update+scale(δ,K) → scale(-1/K)
//   inv:  unscale(-K) → unscale+update(1/K,-δ) → predict(-γ) → update(-β) → predict(-α)

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

    // ---- LOAD: global → shared (identical to forward) ----
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

    // ---- COMPUTE: 5-step INVERSE CDF 9/7 lifting ----
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

    // ---- STORE: shared → global (identical to forward) ----
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

// =============================================================================
// Y/Z-direction INVERSE pass kernel  (synthesis / reconstruction)
// =============================================================================
//
// Identical tile layout and load/store as dwt3d_yz_pass.
// Same boundary clamp rules.  Only the lifting steps are reversed.

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

    // ---- LOAD: global → shared (identical to forward) ----
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

    // ---- COMPUTE: 5-step INVERSE CDF 9/7 lifting along Y/Z ----
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

    // ---- STORE: shared → global (identical to forward) ----
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

}  // namespace wavelet
