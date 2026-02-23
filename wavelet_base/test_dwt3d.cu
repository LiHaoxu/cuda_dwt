// wavelet/test_dwt3d.cu  —  Roundtrip test for 3-D CDF 9/7 DWT + IDWT
//
// Compilation (multi-TU with nvcc):
//   cd wavelet
//   nvcc -std=c++17 -O2 cdf97_3d_host.cu test_dwt3d.cu -o test_dwt3d
//
// Usage:
//   ./test_dwt3d              # run all built-in test cases
//   ./test_dwt3d 128 128 128 3   # custom: NX NY NZ LEVELS

#include "cdf97_3d.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// =============================================================================
// Helpers
// =============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

/// Fill host buffer with deterministic pseudo-random doubles in [-1, 1].
static void FillRandom(double* buf, const std::size_t n, const unsigned seed) {
    std::srand(seed);
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0;
    }
}

/// Compute max absolute error and max relative error between two buffers.
static void CompareBuffers(const double* ref, const double* test,
                           const std::size_t n,
                           double& max_abs, double& max_rel) {
    max_abs = 0.0;
    max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = std::fabs(ref[i] - test[i]);
        if (diff > max_abs) max_abs = diff;
        const double denom = std::fabs(ref[i]);
        if (denom > 1e-15) {
            const double rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
}

// =============================================================================
// Single roundtrip test
// =============================================================================

/// Returns true if the roundtrip error is within tolerance.
static bool RunRoundtrip(const int nx, const int ny, const int nz,
                         const int levels) {
    const std::size_t total = static_cast<std::size_t>(nx) * ny * nz;
    const int ldx = nx;
    const int ldy = ny;

    // Host buffers.
    std::vector<double> h_orig(total);
    std::vector<double> h_result(total);

    FillRandom(h_orig.data(), total, /*seed=*/42u + nx + ny + nz + levels);

    // Device buffer.
    double* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(double)));

    // Copy original data to device.
    CUDA_CHECK(cudaMemcpy(d_data, h_orig.data(), total * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Forward DWT.
    wavelet::dwt3d_forward(d_data, nx, ny, nz, levels, ldx, ldy);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Inverse DWT.
    wavelet::idwt3d_inverse(d_data, nx, ny, nz, levels, ldx, ldy);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back.
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, total * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));

    // Compare.
    double max_abs = 0.0, max_rel = 0.0;
    CompareBuffers(h_orig.data(), h_result.data(), total, max_abs, max_rel);

    // Tolerance: double-precision roundtrip should be ~1e-12 or better.
    constexpr double ABS_TOL = 1e-10;
    const bool pass = (max_abs < ABS_TOL);

    std::printf("  %-20s  nx=%-4d ny=%-4d nz=%-4d levels=%d  "
                "max_abs=%.3e  max_rel=%.3e  %s\n",
                pass ? "[PASS]" : "[FAIL]",
                nx, ny, nz, levels,
                max_abs, max_rel,
                pass ? "" : "*** EXCEEDED TOLERANCE ***");

    return pass;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    // Print device info.
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s\n\n", prop.name);

    if (argc == 5) {
        // Custom test: ./test_dwt3d NX NY NZ LEVELS
        const int nx     = std::atoi(argv[1]);
        const int ny     = std::atoi(argv[2]);
        const int nz     = std::atoi(argv[3]);
        const int levels = std::atoi(argv[4]);

        std::printf("Custom roundtrip test:\n");
        const bool ok = RunRoundtrip(nx, ny, nz, levels);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // Built-in test suite.
    std::printf("Built-in roundtrip tests (DWT → IDWT, compare with original):\n\n");

    int n_pass = 0, n_total = 0;

    // Test various sizes and levels.
    struct TestCase { int nx, ny, nz, levels; };
    const TestCase cases[] = {
        // Small sizes, single level.
        {  8,   8,   8, 1},
        { 16,  16,  16, 1},
        { 32,  32,  32, 1},

        // Medium sizes, multiple levels.
        { 32,  32,  32, 2},
        { 64,  64,  64, 1},
        { 64,  64,  64, 2},
        { 64,  64,  64, 3},

        // Larger sizes.
        {128, 128, 128, 1},
        {128, 128, 128, 2},
        {128, 128, 128, 3},

        // Non-cubic (but still power-of-2 and even).
        { 64,  32,  16, 1},
        { 64,  32,  16, 2},
        {128,  64,  32, 1},
        {128,  64,  32, 2},
        {128,  64,  32, 3},

        // Stress: many levels.
        {256, 256, 256, 1},
        {256, 256, 256, 4},
    };

    for (const auto& tc : cases) {
        ++n_total;
        if (RunRoundtrip(tc.nx, tc.ny, tc.nz, tc.levels)) ++n_pass;
    }

    std::printf("\nSummary: %d / %d passed.\n", n_pass, n_total);

    return (n_pass == n_total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
