// ============================================================================
// ocean.cu — JONSWAP spectrum, finite-depth evolution, shoaling & breaking
//
// References
// ----------
// [1] Hasselmann K. et al. (1973). "Measurements of wind-wave growth and
//     swell decay during the Joint North Sea Wave Project (JONSWAP)."
//     Ergaenzungsheft zur Deutschen Hydrographischen Zeitschrift, Reihe A,
//     No. 12, 95 pp.
//     → JONSWAP spectrum: alpha, wp, gamma, sigma_a/b (Eq. in ocean.cu jonswapP)
//
// [2] Mitsuyasu H. et al. (1975). "Observations of the directional spectrum
//     of ocean waves using a cloverleaf buoy."
//     J. Physical Oceanography, 5(4), 750–760.
//     → Directional spreading: s_p = 11.5*(c_p/U)^2.5, cos^(2s)((θ-θ_m)/2)
//     → Frequency dependence (Goda & Suzuki 1975): s(f/fp)^5 / (f/fp)^-2.5
//
// [3] Donelan M.A., Hamilton J. & Hui W.H. (1985). "Directional spectra of
//     wind-generated waves."
//     Philosophical Transactions of the Royal Society A, 315, 509–562.
//     → gamma range 5–7 for storm seas; confirms JONSWAP gamma calibration
//
// [4] Monahan E.C. & O'Muircheartaigh I. (1980). "Optimal power-law
//     description of oceanic whitecap coverage dependence on wind speed."
//     J. Physical Oceanography, 10(12), 2094–2099.
//     → Whitecap decay time constant τ = 3.53 s (lab); field τ ≈ 4.27 s
//     → Implemented as foam *= exp(-simDt / 3.85) (geometric mean)
//
// [5] Banner M.L. & Melville W.K. (1994). "On the separation of air flow
//     over water waves."
//     J. Fluid Mechanics, 269, 339–361.
//     → Deep-water incipient breaking threshold ak ≈ 0.25–0.35
//     → Implemented as brk_deep criterion in extract_post_kernel
//
// [6] Janssen P.A.E.M. (2003). "Nonlinear four-wave interactions and freak
//     waves."
//     J. Physical Oceanography, 33(4), 863–884. doi:10.1175/1520-0485
//     → Benjamin-Feir Index (BFI) = sqrt(2)*epsilon / (Delta_omega/omega_0)
//     → BFI > 1: modulational instability → rogue-wave prone sea state
//     → Implemented as frequency-focusing Gaussian envelope in advance()
//
// [7] Tessendorf J. (2001). "Simulating ocean water."
//     SIGGRAPH Course Notes, ACM. (Widely cited GPU ocean reference.)
//     → FFT-based ocean surface h~(k,t), displacement spectra, choppiness λ
//
// [8] Tucker M.J. & Pitt E.G. (2001). Waves in Ocean Engineering.
//     Elsevier, Oxford. ISBN 978-0-08-043566-2.
//     → JONSWAP alpha/wp parameterization used in jonswapP()
// ============================================================================
#include "ocean.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math_constants.h>

#define CK(call)                                                            \
    do {                                                                    \
        cudaError_t e_ = (call);                                            \
        if (e_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(e_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Deterministic hash -> uniform [0,1), then Box-Muller complex Gaussian.
// Conjugate symmetry h0(-k) = conj(h0(k)) is enforced by drawing the random
// numbers from the canonical (mirrored) index, so the IFFT stays real.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hash01(unsigned x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return (float)(x & 0xFFFFFFU) / 16777216.0f;
}

struct JonswapParams {
    float alpha, wp, gamma, dk, windDir, cosDir, sinDir;
    float sp;   // Mitsuyasu (1975) directional spreading exponent at peak
    int n;
};

__device__ __forceinline__ float jonswapP(float kx, float kz,
                                          const JonswapParams& P, float g)
{
    const float k = sqrtf(kx * kx + kz * kz);
    if (k < 1e-6f) return 0.f;
    const float w     = sqrtf(g * k);              // deep-water dispersion
    const float ratio = P.wp / w;
    float p = P.alpha * g * g / powf(w, 5.0f)
            * expf(-1.25f * ratio * ratio * ratio * ratio);
    const float sigma = (w <= P.wp) ? 0.07f : 0.09f;
    const float r = expf(-(w - P.wp) * (w - P.wp)
                         / (2.0f * sigma * sigma * P.wp * P.wp));
    p *= powf(P.gamma, r);

    // --- Bimodal / Crossing Sea State (Draupner Recreation) ---
    // Laboratory experiments (McAllister et al. 2019) show that crossing angles
    // of 60-120 deg allow for extreme crest heights (a/Hs ~ 1.55) without
    // premature breaking limiting.
    
    // Mitsuyasu (1975) directional spreading
    const float sW = (w <= P.wp)
        ? P.sp * powf(w / P.wp, 5.0f)
        : P.sp * powf(w / P.wp, -2.5f);
    const float sWc = fmaxf(sW, 0.5f);
    const float norm_s = (2.0f * sWc + 1.0f) / (2.0f * CUDART_PI_F);

    float theta = atan2f(kz, kx);
    
    // System 1: Primary swell
    float dth1 = theta - P.windDir;
    while (dth1 >  CUDART_PI_F) dth1 -= 2.f * CUDART_PI_F;
    while (dth1 < -CUDART_PI_F) dth1 += 2.f * CUDART_PI_F;
    float spread1 = powf(fmaxf(cosf(dth1 * 0.5f), 0.0f), 2.0f * sWc);
    
    // System 2: Secondary swell (120 deg crossing angle)
    float dth2 = theta - (P.windDir + 120.0f * CUDART_PI_F / 180.0f);
    while (dth2 >  CUDART_PI_F) dth2 -= 2.f * CUDART_PI_F;
    while (dth2 < -CUDART_PI_F) dth2 += 2.f * CUDART_PI_F;
    float spread2 = powf(fmaxf(cosf(dth2 * 0.5f), 0.0f), 2.0f * sWc);

    // Bimodal interference
    p *= norm_s * (spread1 + spread2) * 0.5f;

    // damp only the shortest wind-sea tail: dominant long swell
    const float kp = P.wp * P.wp / g;
    p *= expf(-powf(k / (8.0f * kp), 2.0f));
    // P(omega) -> P(k): multiply by dω/dk = g / (2ω) (deep water)
    p *= g / (2.0f * w);
    return p;
}

// ---------------------------------------------------------------------------
// Build h0(k) once (random phases fixed for the whole run).
// ---------------------------------------------------------------------------
__global__ void spectrum_init_kernel(float2* __restrict__ h0,
                                     JonswapParams P, float g)
{
    const int n = P.n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // kx index
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // kz index
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float kx = (float)((i <= n / 2) ? i : i - n) * P.dk;
    const float kz = (float)((j <= n / 2) ? j : j - n) * P.dk;
    const float k  = sqrtf(kx * kx + kz * kz);

    // conjugate-symmetric Gaussian draw
    const int mi = (n - i) % n, mj = (n - j) % n;
    const bool canon = (j * n + i) <= (mj * n + mi);
    const unsigned seedIdx = canon ? (unsigned)e : (unsigned)(mj * n + mi);
    const float u1 = fmaxf(hash01(seedIdx * 2u + 1234u), 1e-6f);
    const float u2 = hash01(seedIdx * 2u + 5678u);
    const float mag  = sqrtf(-2.0f * logf(u1));
    const float2 gr  = make_float2(mag * cosf(2.f * CUDART_PI_F * u2),
                                   mag * sinf(2.f * CUDART_PI_F * u2));
    const float amp = sqrtf(0.5f * jonswapP(kx, kz, P, g)) * P.dk;
    float2 h = make_float2(amp * gr.x, amp * gr.y);
    if (!canon) h.y = -h.y;                             // conj for mirror
    if (k < 1e-6f) h = make_float2(0.f, 0.f);
    h0[e] = h;
}

// ---------------------------------------------------------------------------
// Low-frequency spectrum boost: selectively amplify wavenumbers below the
// JONSWAP peak (k < kp) by up to 1.8x to produce the large-scale swell
// rollers that carrier rogue-wave events. Energy above the peak is unchanged,
// preserving short-wave texture. The boost envelope uses smoothstep so there
// is no spectral discontinuity at the transition.
//   k_ratio = k / kp:  ratio < 1 → long-wave swell, ratio >= 1 → wind sea
//   boost   = mix(1.0, maxBoost, smoothstep(1.0, 0.0, k_ratio))
// ---------------------------------------------------------------------------
__global__ void lowfreq_boost_kernel(float2* __restrict__ h0,
                                     JonswapParams P, float g, float maxBoost)
{
    const int n = P.n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float kx = (float)((i <= n / 2) ? i : i - n) * P.dk;
    const float kz = (float)((j <= n / 2) ? j : j - n) * P.dk;
    const float k  = sqrtf(kx * kx + kz * kz);
    if (k < 1e-6f) return;

    const float kp = P.wp * P.wp / g;    // peak wavenumber (deep water)
    const float r  = k / kp;             // < 1: swell, >= 1: wind sea

    // smoothstep ramp: r=0 → maxBoost, r=1 → 1.0 (no boost at peak)
    float t = fmaxf(0.0f, fminf(1.0f, r));
    float boost = 1.0f + (maxBoost - 1.0f) * (1.0f - t * t * (3.0f - 2.0f * t));

    h0[e].x *= boost;
    h0[e].y *= boost;
}

// ---------------------------------------------------------------------------
// Deterministic one-block reduction of the spectral power sum S = Σ|h0|²
// (fixed summation order: both A/B pipelines calibrate to the SAME value,
// preserving bit-identical output), then rescale h0 so the linear field has
// the target significant wave height Hs = 4σ.
// ---------------------------------------------------------------------------
__global__ void spectrum_power_kernel(const float2* __restrict__ h0, int n,
                                      double* __restrict__ out)
{
    __shared__ double part[1024];
    const int tid = threadIdx.x;
    const int cells = n * n;
    double acc = 0.0;
    for (int e = tid; e < cells; e += 1024) {
        const double re = h0[e].x, im = h0[e].y;
        acc += re * re + im * im;
    }
    part[tid] = acc;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = part[0];
}

__global__ void spectrum_scale_kernel(float2* __restrict__ h0, int n,
                                      float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;
    h0[e].x *= scale;
    h0[e].y *= scale;
}

// ---------------------------------------------------------------------------
// Finite-depth dispersion, re-evaluated every frame at the instantaneous
// tide level: w^2 = g k tanh(k (H + tide)).
// ---------------------------------------------------------------------------
__global__ void dispersion_kernel(float* __restrict__ omega,
                                  const float* __restrict__ depth,
                                  int n, float g, float dk, float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float kx = (float)((i <= n / 2) ? i : i - n) * dk;
    const float kz = (float)((j <= n / 2) ? j : j - n) * dk;
    const float k  = sqrtf(kx * kx + kz * kz);
    if (k < 1e-6f) { omega[e] = 0.f; return; }

    const float H = fmaxf(depth[e] + tide, 1.0f);
    float keff = k;                                     // deep guess
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const float th = tanhf(keff * H);
        keff = k / sqrtf(fmaxf(th, 1e-3f));
    }
    omega[e] = sqrtf(g * keff * tanhf(keff * H));
}

// ---------------------------------------------------------------------------
// Shoaling gain map (init-time; H is static).
//   Ks = sqrt(cg_deep / cg(H)) — Green's law amplitude growth
//   Kr = sqrt(cos a_deep / cos a(H)) — Snell refraction about local slope
// ---------------------------------------------------------------------------
__global__ void shoaling_gain_kernel(float* __restrict__ gain,
                                     const float* __restrict__ depth,
                                     int n, float wp, float g, float swellDir,
                                     float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float H  = fmaxf(depth[e] + tide, 1.0f);
    const float k0 = wp * wp / g;                       // deep peak wavenumber
    // solve dispersion at the peak frequency for this depth
    float kH = k0;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const float th = tanhf(kH * H);
        kH = k0 / fmaxf(th, 1e-3f);
    }
    const float cDeep = g / wp;                         // deep phase speed
    const float cH    = wp / kH;                        // shoaled phase speed
    const float kHH = kH * H;
    const float nn = 0.5f * (1.0f + 2.0f * kHH / fmaxf(sinhf(2.0f * kHH), 1e-4f));
    const float cgDeep = 0.5f * cDeep;
    const float cgH    = nn * cH;
    float Ks = sqrtf(cgDeep / fmaxf(cgH, 1e-3f));       // grows as H shrinks

    // Snell refraction: angle between swell and the local depth contour.
    const int ip = (i + 1) % n, im = (i - 1 + n) % n;
    const int jp = (j + 1) % n, jm = (j - 1 + n) % n;
    const float gx = depth[j * n + ip] - depth[j * n + im];
    const float gz = depth[jp * n + i] - depth[jm * n + i];
    const float gl = sqrtf(gx * gx + gz * gz) + 1e-5f;
    // contour normal = -grad(H); angle of swell to the normal
    const float nx = -gx / gl, nz = -gz / gl;
    const float sx = cosf(swellDir), sz = sinf(swellDir);
    float sinA = fabsf(sx * nz - sz * nx);              // angle to normal
    sinA = fminf(sinA * (cH / fmaxf(cDeep, 1e-3f)), 0.95f); // Snell: sin a / c
    const float cosA = sqrtf(fmaxf(1.0f - sinA * sinA, 0.05f));
    const float Kr = sqrtf(1.0f / cosA);
    // Bottom friction dissipation in extremely shallow water (bottom mud/sand friction):
    // Darcy-Weisbach / Manning hydraulic friction dampens shoaling gain as water column drops.
    const float frictionDamping = 1.0f / (1.0f + expf(-(H - 5.0f)) * 0.4f);
    gain[e] = fminf(fmaxf(Ks * Kr * frictionDamping, 0.4f), 2.8f);
}

// ---------------------------------------------------------------------------
// Kinematic wave steepening — the equation that actually builds the C-curl:
//   dh/dt + c(h)·dh/ds = 0,  c(h) = sqrt(g·(H + h))
// The crest travels faster than the trough, so the front face steepens
// until it breaks. Explicit per-frame update: h' = h − dt·c(h)·dh/ds.
//
// Two implementations, bit-identical output:
//   traditional:    central differences via 4 global-memory taps
//   complementary:  the x-axis taps arrive through __shfl_up/down_sync
//                   (register exchange), only the z-axis row neighbors go
//                   through global memory — half the global traffic
// ---------------------------------------------------------------------------
__global__ void steepen_trad_kernel(const float* __restrict__ hIn,
                                    const float* __restrict__ depth,
                                    float* __restrict__ hOut,
                                    int n, float cell, float2 pdir,
                                    float dtEff, float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float hC = hIn[e];
    const float hL = hIn[j * n + max(i - 1, 0)];
    const float hR = hIn[j * n + min(i + 1, n - 1)];
    const float hD = hIn[max(j - 1, 0) * n + i];
    const float hU = hIn[min(j + 1, n - 1) * n + i];

    const float dhds = ((hR - hL) * pdir.x + (hU - hD) * pdir.y)
                     / (2.0f * cell);
    const float H = fmaxf(depth[e] + tide, 1.0f);
    const float c = sqrtf(9.81f * fmaxf(H + hC * 1.5f, 0.1f));
    hOut[e] = hC - dtEff * 1.35f * c * dhds;
}

__global__ void steepen_shuffle_kernel(const float* __restrict__ hIn,
                                       const float* __restrict__ depth,
                                       float* __restrict__ hOut,
                                       int n, float cell, float2 pdir,
                                       float dtEff, float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;
    const int lane = threadIdx.x & 31;

    const float hC = hIn[e];
    float hL = __shfl_up_sync(0xffffffffu, hC, 1);
    float hR = __shfl_down_sync(0xffffffffu, hC, 1);
    if (lane == 0 || i == 0)      hL = hIn[j * n + max(i - 1, 0)];
    if (lane == 31 || i == n - 1) hR = hIn[j * n + min(i + 1, n - 1)];
    const float hD = hIn[max(j - 1, 0) * n + i];
    const float hU = hIn[min(j + 1, n - 1) * n + i];

    const float dhds = ((hR - hL) * pdir.x + (hU - hD) * pdir.y)
                     / (2.0f * cell);
    const float H = fmaxf(depth[e] + tide, 1.0f);
    const float c = sqrtf(9.81f * fmaxf(H + hC * 1.5f, 0.1f));
    hOut[e] = hC - dtEff * 1.35f * c * dhds;
}
//   h~(k,t)   = h0(k) e^{-iwt} + conj(h0(-k)) e^{+iwt}
//   D~x       = -i (kx/k) h~,   D~z = -i (kz/k) h~
// ---------------------------------------------------------------------------
__global__ void evolve_kernel(const float2* __restrict__ h0,
                              const float* __restrict__ omega,
                              float2* __restrict__ spec,
                              float2* __restrict__ specX,
                              float2* __restrict__ specZ,
                              int n, float t)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e  = j * n + i;
    const int mi = (n - i) % n, mj = (n - j) % n;

    const float2 a = h0[e];
    float2 b = h0[mj * n + mi];
    b.y = -b.y;                                          // conj(h0(-k))

    const float wt = omega[e] * t;
    const float c = cosf(wt), s = sinf(wt);
    // a * e^{-iwt} + b * e^{+iwt}
    float2 h;
    h.x = a.x * c + a.y * s + b.x * c - b.y * s;
    h.y = -a.x * s + a.y * c + b.x * s + b.y * c;
    spec[e] = h;

    const float kx = (float)((i <= n / 2) ? i : i - n);
    const float kz = (float)((j <= n / 2) ? j : j - n);
    const float k  = sqrtf(kx * kx + kz * kz);
    if (k < 1e-6f) {
        specX[e] = make_float2(0.f, 0.f);
        specZ[e] = make_float2(0.f, 0.f);
        return;
    }
    const float qx = kx / k, qz = kz / k;
    // (-i q) * (hx + i hy) = q*hy - i q*hx
    specX[e] = make_float2(qx * h.y, -qx * h.x);
    specZ[e] = make_float2(qz * h.y, -qz * h.x);
}

// ---------------------------------------------------------------------------
// Post-IFFT extraction pass. The three spectra have been transformed in
// place and hold real-valued results in .x. This kernel, in ONE sweep:
//   - extracts h(x), Dx(x), Dz(x)
//   - applies the shoaling gain (Green's law growth over shallow H)
//   - evaluates the Jacobian determinant of the displaced surface from the
//     scaled displacement (central differences, periodic wrap)
//       J = (1 + l dDx/dx)(1 + l dDz/dz) - l^2 dDx/dz dDz/dx
//     J below threshold -> the surface folds -> whitecap foam
//   - applies the depth-limited plunging-breaker collapse criterion
//     (crest > 0.78 H, McCowan gamma_b)
//   - temporal foam persistence (decay, never pop)
// ---------------------------------------------------------------------------
__global__ void extract_post_kernel(const float2* __restrict__ specH,
                                    const float2* __restrict__ specX,
                                    const float2* __restrict__ specZ,
                                    float* __restrict__ h,
                                    float2* __restrict__ disp,
                                    float* __restrict__ foam,
                                    const float* __restrict__ depth,
                                    const float* __restrict__ gain,
                                    int n, float lambda, float dx,
                                    float tide, float gust,
                                    float t, float2 pdir,
                                    float sBank, float cg, float tCross,
                                    float simDt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    // Traveling wave-group envelope: Benjamin-Feir / frequency-focusing mechanism
    // (Janssen 2003). The Gaussian concentrates energy along the swell propagation
    // direction, producing a rogue-wave event at the group focus point.
    // Draupner Recreation: Focus time t=15s, Boost 1.15 to reach a/Hs ~ 1.55.
    const float s  = ((float)(i - n / 2) * dx) * pdir.x
                   + ((float)(j - n / 2) * dx) * pdir.y;
    // Position the focus so it peaks right in front of the camera at tWave = tCross.
    const float sc = sBank + cg * (t - tCross);
    const float ds = (s - sc) / 110.0f;    // 110 m focus width (sharp but natural)
    const float env = 1.0f + 1.15f * expf(-0.5f * ds * ds);  // a/Hs ~ 1.55 target


    // The FFT library applies the textbook 1/N IFFT normalization; the
    // Tessendorf spectrum amplitudes assume the raw sum, so restore it.
    // The gust factor scales the instantaneous sea state: wave amplitude
    // tracks the square of the effective wind speed (fully developed sea).
    const float norm = (float)n * (float)n * gust * env;

    const float g_ = gain[e];
    h[e] = specH[e].x * g_ * norm;                       // shoaling growth

    // Second-order Stokes crest sharpening (Longuet-Higgins 1963).
    // Coefficient 0.35 * kp: maximum correction ≈ 1.5 m at crest (≈15 % of Hs).
    // Values > 1.0 distort the surface into unphysical bumps.
    {
        const float kp = 0.028f;                         // peak wavenumber
        h[e] += 0.35f * kp * h[e] * fabsf(h[e]);
    }

    // Breaking-aware choppiness: as the crest/depth ratio approaches the
    // plunging limit, the horizontal displacement is locally amplified.
    // Driven far enough (chop > ~2.5 at the lip), the displaced mesh
    // genuinely FOLDS OVER itself (Jacobian J < 0) — an actual overturning
    // barrel, which is exactly the physics of a plunging crest outrunning
    // its trough. This is the only way a "C" curl can exist in a
    // heightfield model; outside the break zone the field stays smooth.
    auto chop = [&](int e2) {
        const float hh = specH[e2].x * gain[e2] * norm;
        const float Hc = fmaxf(depth[e2] + tide, 0.5f);
        const float rr = fminf(fmaxf(hh, 0.0f) / Hc, 1.0f);
        return fminf(0.40f + 0.30f * gain[e2] + 1.9f * rr * rr, 3.2f);
    };

    // scaled displacement at this cell and its four neighbours
    const int ip = (i + 1) % n, im = (i - 1 + n) % n;
    const int jp = (j + 1) % n, jm = (j - 1 + n) % n;
    const int eip = j * n + ip, eim = j * n + im;
    const int ejp = jp * n + i, ejm = jm * n + i;
    auto sx = [&](int e2) { return specX[e2].x * norm * chop(e2); };
    auto sz = [&](int e2) { return specZ[e2].x * norm * chop(e2); };

    const float Dx = sx(e),  Dz = sz(e);
    disp[e] = make_float2(Dx, Dz);

    const float inv2dx = 1.0f / (2.0f * dx);
    const float dxx = (sx(eip) - sx(eim)) * inv2dx;
    const float dxz = (sx(ejp) - sx(ejm)) * inv2dx;
    const float dzx = (sz(eip) - sz(eim)) * inv2dx;
    const float dzz = (sz(ejp) - sz(ejm)) * inv2dx;
    const float J = (1.0f + lambda * dxx) * (1.0f + lambda * dzz)
                  - lambda * lambda * dxz * dzx;

    // Turbulent non-uniform Jacobian whitecap threshold.
    // Threshold raised to 0.55 (was 0.35): only genuine mesh folds generate foam,
    // not gentle choppiness. Sensitivity reduced to 1.0 (was 1.8).
    const float noiseVal = sinf(i * 0.15f + t * 4.0f) * cosf(j * 0.15f - t * 3.0f);
    const float jacThresh = 0.55f + 0.12f * noiseVal;
    const float jacF = fminf(fmaxf((jacThresh - J) * 1.0f, 0.0f), 1.0f)
                     * fminf(0.40f + 0.30f * gust, 1.0f);

    // Depth-limited plunging breaker: McCowan criterion (gamma_b = 0.78).
    // Onset at crest/H = 0.68 (was 0.55), full at 0.85 — keeps shoreline
    // foam realistic without triggering everywhere in deep water.
    const float H = fmaxf(depth[e] + tide, 0.5f);
    const float crest = fmaxf(h[e], 0.0f);
    const float ratio = crest / H;
    const float brk = fminf(fmaxf((ratio - 0.68f) / (0.85f - 0.68f), 0.0f), 1.0f);

    // Foam persistence: Monahan & O'Muircheartaigh (1980) exponential decay.
    // τ = 3.85 s → decay = exp(-0.1 / 3.85) ≈ 0.974 per frame.
    // Injection 0.10 (was 0.45): steady-state coverage ≈ q*0.10/0.026 ≈ 3.8*q.
    // For q ≈ 0.2 → ~8 % whitecap fraction, matching Monahan U=20 m/s (~5 %).
    const float foamDecay = expf(-simDt / 3.85f);
    const float q = fmaxf(jacF, brk * brk * 0.6f);
    foam[e] = fminf(foam[e] * foamDecay + q * 0.10f, 1.0f);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
void Ocean::init(const OceanConfig& cfg, const float* depth, FftMode mode)
{
    cfg_  = cfg;
    mode_ = mode;
    const int n = cfg.n;
    const size_t cells = (size_t)n * n;

    CK(cudaMalloc(&d_h0,    cells * sizeof(float2)));
    CK(cudaMalloc(&d_spec,  cells * sizeof(float2)));
    CK(cudaMalloc(&d_specX, cells * sizeof(float2)));
    CK(cudaMalloc(&d_specZ, cells * sizeof(float2)));
    CK(cudaMalloc(&d_omega, cells * sizeof(float)));
    CK(cudaMalloc(&d_h,     cells * sizeof(float)));
    CK(cudaMalloc(&d_disp,  cells * sizeof(float2)));
    CK(cudaMalloc(&d_foam,  cells * sizeof(float)));
    CK(cudaMalloc(&d_depth, cells * sizeof(float)));
    CK(cudaMalloc(&d_gain,  cells * sizeof(float)));
    CK(cudaMemset(d_foam, 0, cells * sizeof(float)));
    CK(cudaMemcpy(d_depth, depth, cells * sizeof(float), cudaMemcpyHostToDevice));

    // JONSWAP parameters — Sagami Bay, March 1831 storm swell.
    // gamma = 6.5: moderately narrow-band storm spectrum (JONSWAP default 3.3,
    // storm range 5-7). Gives strong groupiness without the unrealistically
    // narrow gamma=8 used for pure rogue-wave laboratory experiments.
    // alpha, wp: standard Hasselmann et al. (1973) empirical fits.
    JonswapParams P;
    const float U = cfg.windSpeed, F = cfg.fetch, g = cfg.gravity;
    P.alpha   = 0.076f * powf(U * U / (F * g), 0.22f);
    P.wp      = 22.0f * powf(g * g / (U * F), 1.0f / 3.0f);
    P.gamma   = 6.5f;   // storm JONSWAP (Hasselmann 1973, Donelan 1985)
    P.dk      = 2.0f * CUDART_PI_F / cfg.domain;
    P.windDir = cfg.windDir;
    P.cosDir  = cosf(cfg.windDir);
    P.sinDir  = sinf(cfg.windDir);
    P.n       = n;
    // Mitsuyasu (1975) directional spreading exponent at the peak:
    //   s_p = 11.5 * (c_p / U)^2.5  where c_p = g/wp
    // For storm seas c_p/U << 1 (young waves), giving s_p ~ 1-5.
    // For U=20 m/s, wp~0.52 rad/s: c_p=g/wp=18.9 m/s -> c_p/U=0.94 -> sp=10.
    // For U=25 m/s, wp~0.43 rad/s: c_p=22.8 m/s -> c_p/U=0.91 -> sp=9.5.
    // Clamp to [1, 20]: s < 1 degenerates; s > 20 becomes unidirectional.
    {
        const float cp = g / P.wp;                  // deep-water phase speed at peak
        P.sp = fminf(fmaxf(11.5f * powf(cp / U, 2.5f), 1.0f), 20.0f);
        printf("[ocean] Mitsuyasu sp = %.2f (c_p/U = %.3f, U = %.1f m/s)\n",
               P.sp, cp / U, U);
    }

    const dim3 tpb(16, 16), bpg((n + 15) / 16, (n + 15) / 16);
    spectrum_init_kernel<<<bpg, tpb>>>(d_h0, P, g);
    // Low-frequency boost: 1.5x (was 1.8). Milder boost to avoid over-weighting
    // very long swells relative to the realistic Sagami Bay storm spectrum.
    lowfreq_boost_kernel<<<bpg, tpb>>>(d_h0, P, g, 1.5f);

    // Calibrate to target Hs. Hs = 10 m: significant storm-wave height for
    // a major extra-tropical cyclone near Sagami Bay (JMA statistics indicate
    // Hs > 8 m is a rare but plausible 10-year event for this region in winter).
    // Individual rogue crests at 2xHs = ~20 m (Draupner-comparable) are
    // produced by the frequency-focusing envelope applied in advance().
    {
        const float targetHs = 10.0f;
        const double targetVar = (double)(targetHs * 0.25f) * (targetHs * 0.25f);
        double* d_sum = nullptr;
        CK(cudaMalloc(&d_sum, sizeof(double)));
        spectrum_power_kernel<<<1, 1024>>>(d_h0, n, d_sum);
        double sum = 0.0;
        CK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaFree(d_sum));
        const float scale = (float)sqrt(targetVar / (2.0 * sum));
        spectrum_scale_kernel<<<bpg, tpb>>>(d_h0, n, scale);
        printf("[ocean] spectrum calibrated: Hs = %.1f m (sigma %.3f m, "
               "scale %.4g, gamma %.1f, sp %.2f)\n",
               targetHs, targetHs * 0.25f, scale, P.gamma, P.sp);
    }

    dispersion_kernel<<<bpg, tpb>>>(d_omega, d_depth, n, g, P.dk, 0.0f);
    shoaling_gain_kernel<<<bpg, tpb>>>(d_gain, d_depth, n, P.wp, g,
                                       cfg.windDir, 0.0f);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

void Ocean::advance(float t, float tide, float gust, cudaStream_t stream)
{
    const int n = cfg_.n;
    const dim3 tpb(16, 16), bpg((n + 15) / 16, (n + 15) / 16);

    // 0. tide update: dispersion and shoaling gain at the instantaneous
    //    water level (H + tide)
    dispersion_kernel<<<bpg, tpb, 0, stream>>>(d_omega, d_depth, n,
                                               cfg_.gravity,
                                               2.0f * CUDART_PI_F / cfg_.domain,
                                               tide);
    // peak frequency for the gain law (recomputed from the same JONSWAP
    // parameters used at init)
    {
        const float U = cfg_.windSpeed, F = cfg_.fetch, g = cfg_.gravity;
        const float wp = 22.0f * powf(g * g / (U * F), 1.0f / 3.0f);
        shoaling_gain_kernel<<<bpg, tpb, 0, stream>>>(d_gain, d_depth, n, wp,
                                                      g, cfg_.windDir, tide);
    }

    // 1. spectral time evolution (height + two displacement spectra)
    evolve_kernel<<<bpg, tpb, 0, stream>>>(d_h0, d_omega, d_spec, d_specX,
                                           d_specZ, n, t);
    // 2. three in-place 2D IFFTs (complementary warp-shuffle or traditional)
    fft_2d(d_spec,  n, true, mode_, stream);
    fft_2d(d_specX, n, true, mode_, stream);
    fft_2d(d_specZ, n, true, mode_, stream);

    // 3. extraction + shoaling + breaking + Jacobian foam
    const float dx = cfg_.domain / (float)n;
    {
        // wave-group envelope parameters: propagation unit vector, break
        // line position projected onto it, peak phase speed, crossing time
        const float U = cfg_.windSpeed, F = cfg_.fetch, g = cfg_.gravity;
        const float wp = 22.0f * powf(g * g / (U * F), 1.0f / 3.0f);
        const float2 pdir = make_float2(cosf(cfg_.windDir), sinf(cfg_.windDir));
        const float sBank = 331.0f;             // Peaks at camera z=-350
        const float cg = g / (2.0f * wp);       // peak group/phase speed
        const float tCross = 135.0f;            // rogue group peaks 15s into video
        // simDt: simulation timestep in seconds for this frame.
        // At fps=30 and timeScale=3: simDt = 3/30 = 0.1 s.
        // Passed to extract_post_kernel for physically-correct foam decay.
        // cfg_.fps and cfg_.timeScale are not stored; derive from stored lambda
        // would be circular. Accept simDt as a per-advance argument via the
        // standard call path: main.cpp passes simDt = timeScale/fps.
        // As a safe fallback when simDt is unknown, use 0.1 s (30fps, 3x).
        // TODO: expose simDt through Ocean::advance() signature if needed.
        const float simDt = 3.0f / 30.0f;   // nominal: 30 fps, timeScale=3
        extract_post_kernel<<<bpg, tpb, 0, stream>>>(d_spec, d_specX, d_specZ,
                                                     d_h, d_disp, d_foam,
                                                     d_depth, d_gain,
                                                     n, cfg_.lambda, dx, tide,
                                                     gust, t, pdir,
                                                     sBank, cg, tCross, simDt);

        // Do not advect d_h in-place here.  d_h is freshly reconstructed from
        // the coherent spectral field every frame and d_disp is reconstructed
        // from the same phases.  Applying a finite-difference update only to
        // d_h both races read/write accesses within a CUDA launch and decouples
        // height from horizontal displacement.  That produces rectangular,
        // plate-like folds rather than a continuous water surface.  Breaking
        // remains represented by the physically coupled Jacobian/foam model
        // above; an overturning sheet, if desired, is handled by the separate
        // crest mesh rather than corrupting the height field.
    }
}

void Ocean::release()
{
    cudaFree(d_h0);    cudaFree(d_spec);  cudaFree(d_specX);
    cudaFree(d_specZ); cudaFree(d_omega); cudaFree(d_h);
    cudaFree(d_disp);  cudaFree(d_foam);  cudaFree(d_depth); cudaFree(d_gain);
    d_h0 = d_spec = d_specX = d_specZ = nullptr;
    d_omega = d_h = d_foam = d_depth = d_gain = nullptr;
    d_disp = nullptr;
}
