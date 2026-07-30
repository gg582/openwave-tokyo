// ============================================================================
// fft_gpu.h — 1D batch FFT (two implementations) + 2D IFFT helper
//
//   FFT_COMPLEMENTARY : warp-shuffle antipodal (complementary-pair) radix-2.
//                       The first 5 stages (spans 1..16) run entirely in
//                       registers via __shfl_xor_sync; higher stages are
//                       global-memory butterfly passes.
//   FFT_TRADITIONAL   : classic iterative Cooley-Tukey. Every stage,
//                       including the small spans, is a full global-memory
//                       pass — the control group.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

enum FftMode { FFT_TRADITIONAL = 0, FFT_COMPLEMENTARY = 1 };

// In-place batch 1D FFT on `batch` rows of length `n` (n = 2^k, k >= 5,
// row-major, contiguous). inverse=true computes the IFFT scaled by 1/n.
void batch_fft_1d(float2* data, int n, int batch, bool inverse,
                  FftMode mode, cudaStream_t stream = 0);

// In-place 2D FFT on an n x n row-major matrix (rows FFT, transpose,
// rows FFT, transpose back).
void fft_2d(float2* data, int n, bool inverse, FftMode mode,
            cudaStream_t stream = 0);

// Standalone micro-benchmark: 2D IFFT of size n x n, `iters` iterations.
// Returns milliseconds per iteration for each mode via out params.
void fft_bench(int n, int iters, float* msTraditional, float* msComplementary);
