// ============================================================================
// ocean.cu — JONSWAP spectrum, finite-depth evolution, shoaling & breaking
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
    // cos^2 directional spreading about the wind/swell direction
    const float theta = atan2f(kz, kx);
    const float cd = cosf(theta - P.windDir);
    p *= (2.0f / CUDART_PI_F) * cd * cd;
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
// Finite-depth dispersion, re-evaluated every frame at the instantaneous
// tide level: w^2 = g k tanh(k (H + tide)). A few fixed-point iterations
// recover k_eff(H) so shallow cells slow down as the tide ebbs.
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
    float Kr = sqrtf(1.0f / cosA);

    gain[e] = fminf(fmaxf(Ks * Kr, 0.6f), 2.8f);
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
__global__ void steepen_trad_kernel(const float2* __restrict__ specH,
                                    const float* __restrict__ depth,
                                    float* __restrict__ hOut,
                                    int n, float cell, float2 pdir,
                                    float dtEff, float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    const float hC = specH[e].x;
    const float hL = specH[j * n + max(i - 1, 0)].x;
    const float hR = specH[j * n + min(i + 1, n - 1)].x;
    const float hD = specH[max(j - 1, 0) * n + i].x;
    const float hU = specH[min(j + 1, n - 1) * n + i].x;

    const float dhds = ((hR - hL) * pdir.x + (hU - hD) * pdir.y)
                     / (2.0f * cell);
    const float H = fmaxf(depth[e] + tide, 1.0f);
    const float c = sqrtf(9.81f * fmaxf(H + hC * 1.5f, 0.1f));
    hOut[e] = hC - dtEff * 1.35f * c * dhds;
}

__global__ void steepen_shuffle_kernel(const float2* __restrict__ specH,
                                       const float* __restrict__ depth,
                                       float* __restrict__ hOut,
                                       int n, float cell, float2 pdir,
                                       float dtEff, float tide)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;
    const int lane = threadIdx.x & 31;

    const float hC = specH[e].x;
    // x-axis neighbors through register exchange (identical values to
    // global reads, ~1 cycle, no memory traffic)
    float hL = __shfl_up_sync(0xffffffffu, hC, 1);
    float hR = __shfl_down_sync(0xffffffffu, hC, 1);
    if (lane == 0 || i == 0)      hL = specH[j * n + max(i - 1, 0)].x;
    if (lane == 31 || i == n - 1) hR = specH[j * n + min(i + 1, n - 1)].x;
    // z-axis row neighbors (not expressible via lane shuffle here)
    const float hD = specH[max(j - 1, 0) * n + i].x;
    const float hU = specH[min(j + 1, n - 1) * n + i].x;

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
                                    float sBank, float cg, float tCross)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;

    // Traveling wave-group envelope (real sea groupiness): one dominant
    // set of waves rolling through at the peak phase speed, crossing the
    // break line at mid-clip. Amplitude modulation only — no sculpted
    // shapes.
    const float s  = ((float)(i - n / 2) * dx) * pdir.x
                   + ((float)(j - n / 2) * dx) * pdir.y;
    const float sc = sBank + cg * (t - tCross);
    const float ds = (s - sc) / 150.0f;
    const float env = 1.0f + 0.9f * expf(-0.5f * ds * ds);

    // The FFT library applies the textbook 1/N IFFT normalization; the
    // Tessendorf spectrum amplitudes assume the raw sum, so restore it.
    // The gust factor scales the instantaneous sea state: wave amplitude
    // tracks the square of the effective wind speed (fully developed sea).
    const float norm = (float)n * (float)n * gust * env;

    const float g_ = gain[e];
    h[e] = specH[e].x * g_ * norm;                       // shoaling growth

    // Second-order bound-wave skewness (Stokes-type): real plunging waves
    // are NOT sinusoidal — crests sharpen and troughs flatten. The
    // correction scales with k*a^2: h += skew * kp * h|h|. This is what
    // makes the lip pointed enough to curl tightly instead of mounding.
    {
        const float kp = 0.028f;                         // peak wavenumber
        h[e] += 1.45f * kp * h[e] * fabsf(h[e]);
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

    // Jacobian whitecap: J below threshold -> folding crest; gusts feed
    // extra short-crested wind sea into the whitecap budget
    const float jacF = fminf(fmaxf((0.55f - J) * 2.2f, 0.0f), 1.0f)
                     * fminf(0.55f + 0.45f * gust, 1.25f);

    // Depth-limited plunging breaker: crest height vs instantaneous water
    // depth (local depth + tide). Geometric collapse (McCowan gamma_b).
    const float H = fmaxf(depth[e] + tide, 0.5f);
    const float crest = fmaxf(h[e], 0.0f);
    const float ratio = crest / H;
    const float brk = fminf(fmaxf((ratio - 0.45f) / (0.78f - 0.45f), 0.0f), 1.0f);

    // Air entrainment equation: breaking INJECTS air into the water at
    // rate q; the air fraction A decays as bubbles rise and dissolve.
    // Persistent break zones accumulate a bright lip core (A > 1) while
    // drained regions fade to lacy streaks.
    const float q = fmaxf(jacF, brk * brk);
    foam[e] = fminf(foam[e] * 0.90f + q * 0.8f, 1.5f);
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

    // JONSWAP parameters (Tucker & Pitt 2001 form)
    JonswapParams P;
    const float U = cfg.windSpeed, F = cfg.fetch, g = cfg.gravity;
    P.alpha   = 0.076f * powf(U * U / (F * g), 0.22f);
    P.wp      = 22.0f * powf(g * g / (U * F), 1.0f / 3.0f);
    P.gamma   = 3.3f;
    P.dk      = 2.0f * CUDART_PI_F / cfg.domain;
    P.windDir = cfg.windDir;
    P.cosDir  = cosf(cfg.windDir);
    P.sinDir  = sinf(cfg.windDir);
    P.n       = n;

    const dim3 tpb(16, 16), bpg((n + 15) / 16, (n + 15) / 16);
    spectrum_init_kernel<<<bpg, tpb>>>(d_h0, P, g);
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
        const float sBank = -200.0f * pdir.x + 810.0f * pdir.y;
        const float cg = g / (2.0f * wp);       // peak group/phase speed
        const float tCross = 12.0f;             // set crosses mid-clip
        extract_post_kernel<<<bpg, tpb, 0, stream>>>(d_spec, d_specX, d_specZ,
                                                     d_h, d_disp, d_foam,
                                                     d_depth, d_gain,
                                                     n, cfg_.lambda, dx, tide,
                                                     gust, t, pdir,
                                                     sBank, cg, tCross);
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
