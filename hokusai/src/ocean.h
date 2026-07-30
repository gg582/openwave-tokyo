// ============================================================================
// ocean.h — JONSWAP spectral ocean with bathymetry-driven shoaling
//
// Physics chain per frame (all on GPU):
//   1. h0 spectrum (JONSWAP + cos^2 directional spreading), built once
//   2. time evolution  h~(k,t) = h0(k)e^{-i wt} + conj(h0(-k))e^{+i wt}
//      with finite-depth dispersion w^2 = g k tanh(k H) using the local
//      depth at each spectral cell's georeferenced position
//   3. 2D IFFT (complementary warp-shuffle or traditional control)
//   4. post pass: shoaling amplification Ks(H) = sqrt(cg_deep / cg(H)),
//      Snell refraction factor Kr, depth-limited breaking onset
//      (crest > 0.78 H -> plunging collapse, McCowan criterion), and the
//      Jacobian determinant J of the displaced surface (J < threshold ->
//      whitecap foam), with temporal foam persistence
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include "fft_gpu.h"

struct OceanConfig {
    int   n         = 256;       // spectral grid (n x n)
    float domain    = 4000.0f;   // meters; georeferenced wave patch
    float windSpeed = 18.0f;     // m/s (storm conditions, Hokusai swell)
    float fetch     = 200000.0f; // m
    float windDir   = -0.45f;    // rad; swell out of Sagami-nada (SSE->NNW)
    float gravity   = 9.81f;
    float lambda    = 0.9f;      // choppiness (horizontal displacement)
};

class Ocean {
public:
    // depth: n x n positive meters (row-major, same orientation as the grid)
    void init(const OceanConfig& cfg, const float* depth, FftMode mode);
    // t: wave-clock seconds; tide: instantaneous water level offset (m);
    // gust: wind-gust amplitude multiplier ((U_eff/U_base)^2 envelope)
    void advance(float t, float tide, float gust, cudaStream_t stream = 0);
    void release();

    float*  height() { return d_h; }     // n*n surface elevation (m)
    float2* disp()   { return d_disp; }  // n*n horizontal displacement (m)
    float*  foam()   { return d_foam; }  // n*n foam mask [0,1]
    float*  depth()  { return d_depth; } // n*n bathymetric depth (m)
    float*  gain()   { return d_gain; }  // n*n shoaling gain map
    int     n()      { return cfg_.n; }

private:
    OceanConfig cfg_;
    FftMode     mode_ = FFT_COMPLEMENTARY;

    float2* d_h0    = nullptr;   // base spectrum h0(k)
    float2* d_spec  = nullptr;   // h~(k,t)
    float2* d_specX = nullptr;   // displacement spectra
    float2* d_specZ = nullptr;
    float*  d_omega = nullptr;   // finite-depth dispersion per cell
    float*  d_h     = nullptr;
    float2* d_disp  = nullptr;
    float*  d_foam  = nullptr;
    float*  d_depth = nullptr;
    float*  d_gain  = nullptr;
};
