#include <iostream>

template <int COMPUTE_DIM, int BLOCK_SIZE>
__global__ void kernel(double* input, int ldx, int ldy, int ldz)
{
    const int tid = threadIdx.x;
    static constexpr int BATCH_DIM = 128 / sizeof(double);
    __shared__ double s[BATCH_DIM * COMPUTE_DIM];

    #pragma unroll
    for (int i = 0; i < BATCH_DIM * COMPUTE_DIM / BLOCK_SIZE; ++i)
    {
        const int index = tid + i * BLOCK_SIZE;
        const int coord_in_x = index % COMPUTE_DIM;
        const int coord_in_y = index / COMPUTE_DIM;
        s[index] = input[coord_in_x + coord_in_y * ldx];
    }
}