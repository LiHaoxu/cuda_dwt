

#include "cdf97_3d_kernel.cu"       

#include <algorithm>
#include <cstdio>

namespace wavelet {


static inline int DivCeil(const int a, const int b) {
    return (a + b - 1) / b;
}


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

        // ---- Z pass ----
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



void dwt3d_inverse(double* data,
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

        // ---- Z pass  ----
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

        // ---- Y pass  ----
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

        // ---- X pass  ----
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

}  
