// Float DWT host dispatch — TILE_Y=32, BLOCK_SIZE=512
//
// Overloads the double versions: same function names, float* parameters.
// The compiler selects the right version via C++ overload resolution.

#include "cdf97_3d_f32_kernel.cu"

#include <algorithm>

namespace wavelet {

static inline int DivCeilF(const int a, const int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// dwt3d_forward (float)
// =============================================================================

void dwt3d_forward(float* data,
                   const int nx, const int ny, const int nz,
                   const int levels,
                   const int ldx, const int ldy,
                   cudaStream_t stream) {
    const bool is2d = (nz <= 1);

    for (int lev = 0; lev < levels; ++lev) {
        const int nx_l = nx >> lev;
        const int ny_l = ny >> lev;
        const int nz_l = is2d ? 1 : (nz >> lev);
        const int step = 1 << lev;

        if (nx_l < 2 || ny_l < 2) break;
        if (!is2d && nz_l < 2) break;

        {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(ny_l, TILE_Y_F32),
                            nz_l);
            const dim3 block(BLOCK_SIZE_F32);

            dwt3d_x_pass<COMPUTE_X_F32, TILE_X_F32, TILE_Y_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(ny_l, VALID_PER_TILE_F32),
                            nz_l);
            const dim3 block(BLOCK_SIZE_F32);

            dwt3d_yz_pass<Axis::kY, TILE_X_F32, TILE_Y_F32,
                          VALID_PER_TILE_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        if (!is2d) {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(nz_l, VALID_PER_TILE_F32),
                            ny_l);
            const dim3 block(BLOCK_SIZE_F32);

            dwt3d_yz_pass<Axis::kZ, TILE_X_F32, TILE_Y_F32,
                          VALID_PER_TILE_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }
    }
}

// =============================================================================
// dwt3d_inverse (float)
// =============================================================================

void dwt3d_inverse(float* data,
                   const int nx, const int ny, const int nz,
                   const int levels,
                   const int ldx, const int ldy,
                   cudaStream_t stream) {
    const bool is2d = (nz <= 1);

    for (int lev = levels - 1; lev >= 0; --lev) {
        const int nx_l = nx >> lev;
        const int ny_l = ny >> lev;
        const int nz_l = is2d ? 1 : (nz >> lev);
        const int step = 1 << lev;

        if (nx_l < 2 || ny_l < 2) continue;
        if (!is2d && nz_l < 2) continue;

        if (!is2d) {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(nz_l, VALID_PER_TILE_F32),
                            ny_l);
            const dim3 block(BLOCK_SIZE_F32);

            idwt3d_yz_pass<Axis::kZ, TILE_X_F32, TILE_Y_F32,
                           VALID_PER_TILE_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(ny_l, VALID_PER_TILE_F32),
                            nz_l);
            const dim3 block(BLOCK_SIZE_F32);

            idwt3d_yz_pass<Axis::kY, TILE_X_F32, TILE_Y_F32,
                           VALID_PER_TILE_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        {
            const dim3 grid(DivCeilF(nx_l, COMPUTE_X_F32),
                            DivCeilF(ny_l, TILE_Y_F32),
                            nz_l);
            const dim3 block(BLOCK_SIZE_F32);

            idwt3d_x_pass<COMPUTE_X_F32, TILE_X_F32, TILE_Y_F32, BLOCK_SIZE_F32>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }
    }
}

// =============================================================================
// Padding utilities (float)
// =============================================================================

void dwt3d_reflect_pad(const float* d_src, float* d_dst,
                       int orig_nx, int orig_ny, int orig_nz,
                       int pad_nx, int pad_ny, int pad_nz,
                       cudaStream_t stream) {
    size_t total = size_t(pad_nx) * pad_ny * pad_nz;
    int threads = BLOCK_SIZE_F32;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    kernel_reflect_pad_3d<<<blocks, threads, 0, stream>>>(
        d_src, d_dst, orig_nx, orig_ny, orig_nz, pad_nx, pad_ny, pad_nz);
}

void dwt3d_crop(const float* d_padded, float* d_dst,
                int orig_nx, int orig_ny, int orig_nz,
                int pad_nx, int pad_ny, int pad_nz,
                cudaStream_t stream) {
    size_t total = size_t(orig_nx) * orig_ny * orig_nz;
    int threads = BLOCK_SIZE_F32;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    kernel_crop_3d<<<blocks, threads, 0, stream>>>(
        d_padded, d_dst, orig_nx, orig_ny, orig_nz, pad_nx, pad_ny, pad_nz);
}

}  // namespace wavelet
