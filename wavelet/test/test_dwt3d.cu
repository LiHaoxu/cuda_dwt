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
/// Handles arbitrary (non-power-of-2) dimensions by padding to multiples of
/// 2^levels with symmetric reflection, then cropping back after IDWT.
/// Error is evaluated only on the original (unpadded) region.
static bool RunRoundtrip(const int nx, const int ny, const int nz,
                         const int levels) {
    const bool is2d = (nz <= 1);
    const std::size_t total = static_cast<std::size_t>(nx) * ny * std::max(nz, 1);

    // Compute padded dimensions (multiples of 2^levels).
    const int pnx = wavelet::dwt3d_padded_dim(nx, levels);
    const int pny = wavelet::dwt3d_padded_dim(ny, levels);
    const int pnz = is2d ? 1 : wavelet::dwt3d_padded_dim(nz, levels);
    const bool needs_pad = (pnx != nx || pny != ny || pnz != nz);
    const std::size_t pad_total = static_cast<std::size_t>(pnx) * pny * pnz;

    // Host buffers.
    std::vector<double> h_orig(total);
    std::vector<double> h_result(total);

    FillRandom(h_orig.data(), total, /*seed=*/42u + nx + ny + nz + levels);

    // Device buffers.
    double* d_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_orig, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_orig, h_orig.data(), total * sizeof(double),
                          cudaMemcpyHostToDevice));

    double* d_data = nullptr;
    if (needs_pad) {
        CUDA_CHECK(cudaMalloc(&d_data, pad_total * sizeof(double)));
        wavelet::dwt3d_reflect_pad(d_orig, d_data,
                                   nx, ny, std::max(nz, 1),
                                   pnx, pny, pnz);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaMalloc(&d_data, pad_total * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, d_orig, total * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    // Forward DWT on padded data.
    wavelet::dwt3d_forward(d_data, pnx, pny, pnz, levels, pnx, pny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Inverse DWT on padded data.
    wavelet::dwt3d_inverse(d_data, pnx, pny, pnz, levels, pnx, pny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Crop back to original dimensions.
    if (needs_pad) {
        double* d_cropped = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cropped, total * sizeof(double)));
        wavelet::dwt3d_crop(d_data, d_cropped,
                            nx, ny, std::max(nz, 1),
                            pnx, pny, pnz);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_cropped,
                              total * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cropped));
    } else {
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_data,
                              total * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_orig));

    // Compare only the original region (not padded area).
    double max_abs = 0.0, max_rel = 0.0;
    CompareBuffers(h_orig.data(), h_result.data(), total, max_abs, max_rel);

    constexpr double ABS_TOL = 1e-10;
    const bool pass = (max_abs < ABS_TOL);

    std::printf("  %-20s  nx=%-4d ny=%-4d nz=%-4d  pad=%dx%dx%d  levels=%d  "
                "max_abs=%.3e  max_rel=%.3e  %s\n",
                pass ? "[PASS]" : "[FAIL]",
                nx, ny, nz, pnx, pny, pnz, levels,
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
        // Power-of-2 sizes (classic)
        {256, 256, 256, 5},
        {512, 512, 512, 6},
        // Non-power-of-2 sizes (exercise padding logic)
        {100, 100, 100, 3},
        {200, 150, 180, 4},
        {300, 300, 1, 3},       // 2D non-pow2
        {129, 65, 33, 2},
        {80, 60, 40, 3},
    };

    for (const auto& tc : cases) {
        ++n_total;
        if (RunRoundtrip(tc.nx, tc.ny, tc.nz, tc.levels)) ++n_pass;
    }

    std::printf("\nSummary: %d / %d passed.\n", n_pass, n_total);

    return (n_pass == n_total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
