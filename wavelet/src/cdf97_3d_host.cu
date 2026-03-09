// wavelet/cdf97_3d_host.cu  —  Multilevel dispatch for 3-D CDF 9/7 DWT
//
// Compilation (single-TU approach — kernel templates are included):
//   nvcc -std=c++17 -O2 cdf97_3d_host.cu -o dwt3d_test
//
// The file #includes the kernel source so that template bodies are visible
// at the <<<>>> launch sites without requiring CUDA separate compilation.

#include "cdf97_3d_kernel.cu"  // brings in cdf97_3d.cuh transitively

#include <algorithm>
#include <cstdio>

namespace wavelet {

// ---- Utility: integer ceiling division ----
static inline int DivCeil(const int a, const int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// dwt3d_forward  —  public entry point
// =============================================================================

void dwt3d_forward(double* data,
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

        // ---- X pass ----
        {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(ny_l, TILE_Y),
                            nz_l);
            const dim3 block(BLOCK_SIZE);

            dwt3d_x_pass<COMPUTE_X, TILE_X, TILE_Y, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        // ---- Y pass ----
        {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(ny_l, VALID_PER_TILE),
                            nz_l);
            const dim3 block(BLOCK_SIZE);

            dwt3d_yz_pass<Axis::kY, TILE_X, TILE_Y,
                          VALID_PER_TILE, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        // ---- Z pass (skip entirely for 2D data) ----
        if (!is2d) {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(nz_l, VALID_PER_TILE),
                            ny_l);
            const dim3 block(BLOCK_SIZE);

            dwt3d_yz_pass<Axis::kZ, TILE_X, TILE_Y,
                          VALID_PER_TILE, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }
    }
}

// =============================================================================
// dwt3d_inverse  —  public entry point (synthesis / reconstruction)
// =============================================================================

void dwt3d_inverse(double* data,
                   const int nx, const int ny, const int nz,
                   const int levels,
                   const int ldx, const int ldy,
                   cudaStream_t stream) {
    // Inverse processes levels in reverse order.
    // Within each level the axis order is reversed: Z → Y → X.
    const bool is2d = (nz <= 1);

    for (int lev = levels - 1; lev >= 0; --lev) {
        const int nx_l = nx >> lev;
        const int ny_l = ny >> lev;
        const int nz_l = is2d ? 1 : (nz >> lev);
        const int step = 1 << lev;

        if (nx_l < 2 || ny_l < 2) continue;
        if (!is2d && nz_l < 2) continue;

        // ---- Z pass (inverse, skip entirely for 2D data) ----
        if (!is2d) {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(nz_l, VALID_PER_TILE),
                            ny_l);
            const dim3 block(BLOCK_SIZE);

            idwt3d_yz_pass<Axis::kZ, TILE_X, TILE_Y,
                           VALID_PER_TILE, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        // ---- Y pass (inverse) ----
        {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(ny_l, VALID_PER_TILE),
                            nz_l);
            const dim3 block(BLOCK_SIZE);

            idwt3d_yz_pass<Axis::kY, TILE_X, TILE_Y,
                           VALID_PER_TILE, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }

        // ---- X pass (inverse) ----
        {
            const dim3 grid(DivCeil(nx_l, COMPUTE_X),
                            DivCeil(ny_l, TILE_Y),
                            nz_l);
            const dim3 block(BLOCK_SIZE);

            idwt3d_x_pass<COMPUTE_X, TILE_X, TILE_Y, BLOCK_SIZE>
                <<<grid, block, 0, stream>>>(
                    data, nx_l, ny_l, nz_l, step, ldx, ldy);
        }
    }
}

// =============================================================================
// Padding utilities
// =============================================================================

int dwt3d_padded_dim(int dim, int levels) {
    if (dim <= 1) return dim;
    int block = 1 << levels;
    return ((dim + block - 1) / block) * block;
}

void dwt3d_reflect_pad(const double* d_src, double* d_dst,
                       int orig_nx, int orig_ny, int orig_nz,
                       int pad_nx, int pad_ny, int pad_nz,
                       cudaStream_t stream) {
    size_t total = size_t(pad_nx) * pad_ny * pad_nz;
    int threads = BLOCK_SIZE;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    kernel_reflect_pad_3d<<<blocks, threads, 0, stream>>>(
        d_src, d_dst, orig_nx, orig_ny, orig_nz, pad_nx, pad_ny, pad_nz);
}

void dwt3d_crop(const double* d_padded, double* d_dst,
                int orig_nx, int orig_ny, int orig_nz,
                int pad_nx, int pad_ny, int pad_nz,
                cudaStream_t stream) {
    size_t total = size_t(orig_nx) * orig_ny * orig_nz;
    int threads = BLOCK_SIZE;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    kernel_crop_3d<<<blocks, threads, 0, stream>>>(
        d_padded, d_dst, orig_nx, orig_ny, orig_nz, pad_nx, pad_ny, pad_nz);
}

}  // namespace wavelet
