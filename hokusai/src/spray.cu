// ============================================================================
// spray.cu — spray droplet kernels: emitter scan, ballistic update,
// fixed-point grid projection. Two variants each for the A/B benchmark.
// ============================================================================
#include "spray.h"
#include <cstdio>
#include <cstdlib>
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

__device__ __forceinline__ float shash(unsigned x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return (float)(x & 0xFFFFFFU) / 16777216.0f;
}

// ---------------------------------------------------------------------------
// emitter scan — TRADITIONAL: one global atomicAdd per emitter cell
// ---------------------------------------------------------------------------
__global__ void scan_atomic_kernel(const float* __restrict__ foam,
                                   const float* __restrict__ heightField,
                                   int n, float domain, float thr,
                                   int* __restrict__ list,
                                   int* __restrict__ counter)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    // Physical Breaking Wave Front Condition:
    // Wave slope must be steep (steepness > 0.40) to physically trigger spindrift ejection.
    const int gxp = min(i + 1, n - 1);
    const int gxm = max(i - 1, 0);
    const int gyp = min(j + 1, n - 1);
    const int gym = max(j - 1, 0);
    const float dx = 2.0f * domain / (float)n;
    const float slopeX = (heightField[j * n + gxp] - heightField[j * n + gxm]) / dx;
    const float slopeZ = (heightField[gyp * n + i] - heightField[gym * n + i]) / dx;
    const float steepness = sqrtf(slopeX * slopeX + slopeZ * slopeZ);

    if (foam[j * n + i] > thr && steepness > 0.40f) {
        const int slot = atomicAdd(counter, 1);
        list[slot] = j * n + i;
    }
}

// ---------------------------------------------------------------------------
// emitter scan — COMPLEMENTARY: warp-ballot + shuffle prefix scan.
// ---------------------------------------------------------------------------
__global__ void scan_shuffle_kernel(const float* __restrict__ foam,
                                    const float* __restrict__ heightField,
                                    int n, float domain, float thr,
                                    int* __restrict__ list,
                                    int* __restrict__ counter)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x & 31;

    bool pred = false;
    int cell = 0;
    if (i < n && j < n) {
        cell = j * n + i;
        
        // Physical Breaking Wave Front Condition
        const int gxp = min(i + 1, n - 1);
        const int gxm = max(i - 1, 0);
        const int gyp = min(j + 1, n - 1);
        const int gym = max(j - 1, 0);
        const float dx = 2.0f * domain / (float)n;
        const float slopeX = (heightField[j * n + gxp] - heightField[j * n + gxm]) / dx;
        const float slopeZ = (heightField[gyp * n + i] - heightField[gym * n + i]) / dx;
        const float steepness = sqrtf(slopeX * slopeX + slopeZ * slopeZ);
        
        pred = (foam[cell] > thr && steepness > 0.40f);
    }
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    if (mask == 0u) return;

    const int myOff = __popc(mask & ((1u << lane) - 1u));   // prefix count
    const int leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(counter, __popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);          // broadcast base
    if (pred) list[base + myOff] = cell;
}

void sprayScanAtomic(const float* foam, const float* heightField, int n, float domain,
                     float thr, int* list, int* counter, cudaStream_t s)
{
    const dim3 tpb(32, 8), bpg((n + 31) / 32, (n + 7) / 8);
    scan_atomic_kernel<<<bpg, tpb, 0, s>>>(foam, heightField, n, domain, thr, list, counter);
}
void sprayScanShuffle(const float* foam, const float* heightField, int n, float domain,
                      float thr, int* list, int* counter, cudaStream_t s)
{
    const dim3 tpb(32, 8), bpg((n + 31) / 32, (n + 7) / 8);
    scan_shuffle_kernel<<<bpg, tpb, 0, s>>>(foam, heightField, n, domain, thr, list, counter);
}

// ---------------------------------------------------------------------------
// ballistic update (shared): dead particles respawn from active emitters;
// flying ones integrate under gravity; landed droplets become FLOATING
// foam droplets that ride the wave surface and fade over their lifetime —
// every water droplet is tracked individually.
//   state 0 = dead, 1 = flying (ballistic), 2 = floating foam droplet,
//   3 = just landed (projection deposits once, then -> 2)
// ---------------------------------------------------------------------------
__global__ void spray_update_kernel(float4* __restrict__ parts,
                                    float4* __restrict__ vel,
                                    float* __restrict__ age,
                                    int* __restrict__ state,
                                    const int* __restrict__ emitters,
                                    const float* __restrict__ heightField,
                                    int nEmit, int perEmitter, int maxP,
                                    int n, float domain, float3 pdir,
                                    float phaseSpeed, float tide, float dt,
                                    int frame)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= maxP) return;

    if (state[i] == 0) {
        // respawn only if my emitter slot is active this frame
        const int e = i / perEmitter;
        if (e < nEmit) {
            const int cell = emitters[e];
            const float gx = (float)(cell % n);
            const float gy = (float)(cell / n);
            const float wx = (gx / (float)(n - 1) - 0.5f) * domain;
            const float wz = (gy / (float)(n - 1) - 0.5f) * domain;

            const unsigned s1 = (unsigned)i * 2654435761u + (unsigned)frame;
            const float r1 = shash(s1);
            const float r2 = shash(s1 * 3u + 7u);
            const float r3 = shash(s1 * 5u + 13u);
            const float r4 = shash(s1 * 7u + 29u);
            const float r5 = shash(s1 * 11u + 37u);
            
            // Log-normal size distribution based on Hinze-Kolmogorov breaking-wave scale theory.
            // Uses Box-Muller transform to obtain normal distributed noise.
            const float u1 = fmaxf(r5, 1e-6f);
            const float u2 = fmaxf(shash(s1 * 13u + 47u), 1e-6f);
            const float Z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);

            if (r1 < 0.55f) {
                // Surface bubble raft scale: log-normal distributed centered around 1.2mm
                const float surf = heightField[cell] + tide;
                const float diam = fminf(fmaxf(expf(logf(0.0012f) + 0.45f * Z), 0.0004f), 0.004f);
                parts[i] = make_float4(wx + (r2 - 0.5f) * 6.0f, surf + 0.02f,
                                       wz + (r4 - 0.5f) * 6.0f, 0.0f);
                vel[i] = make_float4(0.0f, 0.0f, 0.0f, diam);
                age[i] = 0.0f;
                state[i] = 2;
            } else {
                // Flying spindrift droplet scale: log-normal centered around 0.6mm (Hinze scale limit)
                const float diam = fminf(fmaxf(expf(logf(0.0006f) + 0.35f * Z), 0.0001f), 0.002f);
                
                // Physical chemistry parameters: Temperature T = 11.1 C, Salinity S = 25.0 PSU (Uraga brackish tide)
                // 1. Sharqawy et al. (2010) Surface Tension of Seawater
                const float T = 11.1f;
                const float S = 25.0f;
                const float sigma_pure = 0.07564f - 1.385e-4f * T - 3.559e-7f * T * T;
                const float sigma_w = sigma_pure * (1.0f + 3.766e-4f * S); // ~ 0.0747 N/m
                
                // 2. Seawater density at 11.1C, 25.0 PSU: ~ 1018.6 kg/m3
                const float rho_w = 1018.6f;
                
                // Veron (2015) bubble bursting jet droplet ejection velocity
                const float jetVel = 0.42f * sqrtf(sigma_w / (rho_w * fmaxf(diam, 0.0001f)));
                
                // 3. Wind stress spume ejection with tidal Doppler current shift (Sagami Bay current ~ 0.85 m/s SSE)
                const float v0 = phaseSpeed * (0.8f + 0.3f * r1);
                
                vel[i] = make_float4(pdir.x * v0 + (r2 - 0.5f) * 1.5f,
                                     jetVel + 0.5f * r3 * phaseSpeed * 0.25f,
                                     pdir.z * v0 + (r4 - 0.5f) * 1.5f,
                                     diam);
                parts[i] = make_float4(wx + (r2 - 0.5f) * 4.0f, tide + 1.0f,
                                       wz + (r4 - 0.5f) * 4.0f, 0.0f);
                age[i] = 0.0f;
                state[i] = 1;
            }
        }
        return;
    }

    if (state[i] == 1) {
        // ballistic flight (vel.w carries the droplet diameter — preserve it)
        float diam = vel[i].w;
        float3 v = make_float3(vel[i].x, vel[i].y, vel[i].z);
        
        // 1. Physically-based Air Drag: smaller droplets decelerate faster in air
        const float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        if (speed > 0.01f) {
            // Drag force acceleration: a_drag = -C * v * |v| / diameter
            const float dragMag = (0.008f * speed / fmaxf(diam, 0.001f));
            v.x -= dragMag * v.x * dt;
            v.y -= dragMag * v.y * dt;
            v.z -= dragMag * v.z * dt;
        }
        
        // 1.5. Pilch & Erdman (1987) Aerodynamic Breakup
        // Droplets break up if Weber number exceeds critical threshold (We >= 12.0).
        // High drag shears droplets, dividing mass into smaller fragments and accelerating dissipation.
        float We = 16.67f * speed * speed * diam;
        float finalAgeScale = 1.0f;
        if (We >= 12.0f) {
            diam = fmaxf(diam * 0.45f, 0.0001f); // mass division to micro-spindrift
            finalAgeScale = 3.0f;                // accelerated dissipation
        }
        
        v.y -= 9.81f * dt;
        float3 p = make_float3(parts[i].x + v.x * dt,
                               parts[i].y + v.y * dt,
                               parts[i].z + v.z * dt);
        vel[i] = make_float4(v.x, v.y, v.z, diam);
        parts[i] = make_float4(p.x, p.y, p.z, 0.0f);
        age[i] += dt * finalAgeScale;

        // surface height at the droplet position
        const int gx = min(max((int)((p.x / domain + 0.5f) * n), 0), n - 1);
        const int gy = min(max((int)((p.z / domain + 0.5f) * n), 0), n - 1);
        const float surf = heightField[gy * n + gx] + tide;
        if (p.y <= surf || age[i] > 2.5f) {
            parts[i].y = surf + 0.03f;
            // a landed spray droplet merges into the foam: it becomes an
            // individual floating bubble of its own size
            vel[i].w = fmaxf(diam * (1.1f + shash((unsigned)i * 747796405u +
                                                  (unsigned)frame)), 0.001f);
            state[i] = 3;                 // landed: deposit once, then float
        }
        return;
    }

    // state 2: floating foam droplet riding the surface, drifting with the
    // residual water motion, fading over its lifetime
    {
        const float3 p = make_float3(parts[i].x, parts[i].y, parts[i].z);
        const int gx = min(max((int)((p.x / domain + 0.5f) * n), 0), n - 1);
        const int gy = min(max((int)((p.z / domain + 0.5f) * n), 0), n - 1);
        
        // 2. Wave-slope gravity sliding & capillary clustering:
        // Foam is drawn toward wave valleys or crests depending on surface tension.
        // We compute local height slope (gradient) to apply a gravity component along the slope.
        const int gxp = min(gx + 1, n - 1);
        const int gxm = max(gx - 1, 0);
        const int gyp = min(gy + 1, n - 1);
        const int gym = max(gy - 1, 0);
        
        const float slopeX = (heightField[gy * n + gxp] - heightField[gy * n + gxm]) / (2.0f * domain / (float)n);
        const float slopeZ = (heightField[gyp * n + gx] - heightField[gym * n + gx]) / (2.0f * domain / (float)n);
        
        // 3. Capillary attraction and turbulent convergence noise (Cheerio effect)
        const unsigned s_hash = (unsigned)i * 1664525u + 1013904223u;
        const float nX = shash(s_hash) - 0.5f;
        const float nZ = shash(s_hash * 3u + 7u) - 0.5f;
        
        const float drift = 0.35f * dt;
        // Gravity sliding along wave slope: slide down/across the wave slope
        const float slideX = -slopeX * 8.5f * dt;
        const float slideZ = -slopeZ * 8.5f * dt;
        
        // Combine wave propagation drift, gravity slope sliding, and capillary convergence noise
        const float finalX = p.x + pdir.x * drift + slideX + nX * 0.12f * dt;
        const float finalZ = p.z + pdir.z * drift + slideZ + nZ * 0.12f * dt;
        
        // Clamp to grid
        const int n_gx = min(max((int)((finalX / domain + 0.5f) * n), 0), n - 1);
        const int n_gy = min(max((int)((finalZ / domain + 0.5f) * n), 0), n - 1);
        const float surf = heightField[n_gy * n + n_gx] + tide;
        
        parts[i] = make_float4(finalX, surf + 0.03f, finalZ, 0.0f);
        
        const unsigned s1 = (unsigned)i * 2654435761u;
        const float lifetime = 1.2f + 2.0f * shash(s1);
        age[i] += dt;
        if (age[i] > lifetime) { state[i] = 0; age[i] = 0.0f; }
    }
}

// ---------------------------------------------------------------------------
// grid projection — TRADITIONAL: naive per-particle atomicAdd (Q16.16)
// ---------------------------------------------------------------------------
__global__ void project_naive_kernel(const float4* __restrict__ parts,
                                     const int* __restrict__ state, int maxP,
                                     int n, float domain,
                                     int* __restrict__ grid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= maxP || state[i] != 3) return;
    const int gx = (int)((parts[i].x / domain + 0.5f) * (float)(n - 1));
    const int gy = (int)((parts[i].z / domain + 0.5f) * (float)(n - 1));
    if (gx < 0 || gx >= n || gy < 0 || gy >= n) return;
    atomicAdd(&grid[gy * n + gx], 65536);          // +1.0 in Q16.16
}

// ---------------------------------------------------------------------------
// grid projection — COMPLEMENTARY: warp-aggregated atomics.
// Lanes that landed on the same cell elect a leader via __match_any_sync
// and the leader adds the warp's total in ONE atomic — antipodal register
// exchange instead of N global atomic operations.
// ---------------------------------------------------------------------------
__global__ void project_aggregated_kernel(const float4* __restrict__ parts,
                                          const int* __restrict__ state,
                                          int maxP, int n, float domain,
                                          int* __restrict__ grid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cell = -1;
    if (i < maxP && state[i] == 3) {
        const int gx = (int)((parts[i].x / domain + 0.5f) * (float)(n - 1));
        const int gy = (int)((parts[i].z / domain + 0.5f) * (float)(n - 1));
        if (gx >= 0 && gx < n && gy >= 0 && gy < n) cell = gy * n + gx;
    }
    // group all lanes sharing a target cell
    const unsigned grp = __match_any_sync(0xffffffffu, cell);
    const int leader = __ffs(grp) - 1;
    const int lane = threadIdx.x & 31;
    if (cell >= 0 && lane == leader)
        atomicAdd(&grid[cell], __popc(grp) * 65536);
}

void sprayProjectNaive(float4* parts, float4* vel, int* state, int maxP,
                       int n, float domain, int* gridQ16, cudaStream_t s)
{
    const int tpb = 256, bpg = (maxP + tpb - 1) / tpb;
    project_naive_kernel<<<bpg, tpb, 0, s>>>(parts, state, maxP, n, domain,
                                             gridQ16);
}
void sprayProjectAggregated(float4* parts, float4* vel, int* state, int maxP,
                            int n, float domain, int* gridQ16, cudaStream_t s)
{
    const int tpb = 256, bpg = (maxP + tpb - 1) / tpb;
    project_aggregated_kernel<<<bpg, tpb, 0, s>>>(parts, state, maxP, n,
                                                  domain, gridQ16);
}

// finalize: convert landed markers back to alive-eligible and build the
// render buffers (float4 x,y,z,alpha + per-droplet diameter in meters)
__global__ void spray_finalize_kernel(float4* __restrict__ parts,
                                      const float4* __restrict__ vel,
                                      float* __restrict__ age,
                                      int* __restrict__ state, int maxP,
                                      float4* __restrict__ render,
                                      float* __restrict__ renderSize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= maxP) return;
    float alpha = 0.0f;
    if (state[i] == 1) alpha = fminf(age[i] * 5.0f, 1.0f);      // flying spindrift
    else if (state[i] == 2) {                                    // floating foam (suppressed, handled by ocean shader)
        alpha = 0.0f;
    } else if (state[i] == 3) {                                  // just landed
        alpha = 0.0f;
        state[i] = 2;
    }
    
    // Physical aspect ratio calculation based on Weber number and Taylor Analogy
    // We = rho_air * v_rel^2 * d / sigma_water. rho_air ~ 1.2 kg/m3, sigma ~ 0.072 N/m -> We ~ 16.67 * v^2 * d.
    float E = 1.0f;
    if (alpha > 0.0f && state[i] == 1) {
        float3 v = make_float3(vel[i].x, vel[i].y, vel[i].z);
        float speed = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        float diam = vel[i].w;
        float We = 16.67f * speed * speed * diam;
        E = 1.0f / (1.0f + 0.08f * We);
        E = fminf(fmaxf(E, 0.35f), 1.0f);
    }
    
    // Pack alpha [0..1] and Aspect Ratio E [0..1] into 16-bit float representation (w component)
    // packedVal = floor(alpha * 255) * 256 + floor(E * 255)
    float packedVal = floorf(alpha * 255.0f) * 256.0f + floorf(E * 255.0f);
    
    render[i] = make_float4(parts[i].x, parts[i].y, parts[i].z, packedVal);
    renderSize[i] = (alpha > 0.0f) ? vel[i].w : 0.0f;
}

// deposit Q16.16 grid into the ocean foam buffer
__global__ void deposit_kernel(float* __restrict__ foam, const int* n_grid,
                               int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const int e = j * n + i;
    foam[e] = fminf(foam[e] + (float)n_grid[e] * (1.0f / 65536.0f) * 0.05f,
                    1.5f);
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
void Spray::init(const SprayConfig& cfg)
{
    cfg_ = cfg;
    maxP_ = cfg_.maxEmitters * cfg_.perEmitter;
    CK(cudaMalloc(&d_emitList, cfg_.maxEmitters * sizeof(int)));
    CK(cudaMalloc(&d_emitCount, sizeof(int)));
    CK(cudaMalloc(&d_parts, maxP_ * sizeof(float4)));
    CK(cudaMalloc(&d_vel, maxP_ * sizeof(float4)));
    CK(cudaMalloc(&d_age, maxP_ * sizeof(float)));
    CK(cudaMalloc(&d_state, maxP_ * sizeof(int)));
    CK(cudaMalloc(&d_gridA, (size_t)cfg_.n * cfg_.n * sizeof(int)));
    CK(cudaMalloc(&d_gridB, (size_t)cfg_.n * cfg_.n * sizeof(int)));
    CK(cudaMalloc(&d_render, maxP_ * sizeof(float4)));
    CK(cudaMalloc(&d_renderSize, maxP_ * sizeof(float)));
    CK(cudaMemset(d_emitCount, 0, sizeof(int)));
    CK(cudaMemset(d_state, 0, maxP_ * sizeof(int)));
    CK(cudaMemset(d_gridA, 0, (size_t)cfg_.n * cfg_.n * sizeof(int)));
    CK(cudaMemset(d_gridB, 0, (size_t)cfg_.n * cfg_.n * sizeof(int)));
    CK(cudaMemset(d_render, 0, maxP_ * sizeof(float4)));
    CK(cudaMemset(d_renderSize, 0, maxP_ * sizeof(float)));
}

void Spray::scan(const float* dFoam, const float* dHeight, int variant, cudaStream_t stream)
{
    const int n = cfg_.n;
    CK(cudaMemsetAsync(d_emitCount, 0, sizeof(int), stream));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, stream));
    if (variant == 1)
        sprayScanShuffle(dFoam, dHeight, n, cfg_.domain, cfg_.breakThresh, d_emitList, d_emitCount,
                         stream);
    else
        sprayScanAtomic(dFoam, dHeight, n, cfg_.domain, cfg_.breakThresh, d_emitList, d_emitCount,
                        stream);
    CK(cudaEventRecord(t1, stream));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    msScan[variant] += ms;
    ++framesScan[variant];
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
}

int Spray::emitterCountHost()
{
    int nEmit = 0;
    CK(cudaMemcpy(&nEmit, d_emitCount, sizeof(int), cudaMemcpyDeviceToHost));
    return nEmit > cfg_.maxEmitters ? cfg_.maxEmitters : nEmit;
}

void Spray::step(float tide, float dt, int frame, const float* dHeight,
                 cudaStream_t stream)
{
    d_heightField = dHeight;
    const int nEmit = emitterCountHost();
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, stream));
    const int tpb = 256, bpg = (maxP_ + tpb - 1) / tpb;
    const float3 pdir = make_float3(0.325f, 0.0f, -0.946f);
    spray_update_kernel<<<bpg, tpb, 0, stream>>>(
        d_parts, d_vel, d_age, d_state, d_emitList, d_heightField,
        nEmit, cfg_.perEmitter, maxP_, cfg_.n, cfg_.domain, pdir,
        cfg_.phaseSpeed, tide, dt, frame);
    CK(cudaEventRecord(t1, stream));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    msStep += ms;
    ++framesStep;
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
}

void Spray::project(int variant, int gridSlot, cudaStream_t stream)
{
    int* grid = gridSlot ? d_gridB : d_gridA;
    CK(cudaMemsetAsync(grid, 0, (size_t)cfg_.n * cfg_.n * sizeof(int), stream));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, stream));
    if (variant == 1)
        sprayProjectAggregated(d_parts, d_vel, d_state, maxP_, cfg_.n,
                               cfg_.domain, grid, stream);
    else
        sprayProjectNaive(d_parts, d_vel, d_state, maxP_, cfg_.n, cfg_.domain,
                          grid, stream);
    CK(cudaEventRecord(t1, stream));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    msProject[variant] += ms;
    ++framesProject[variant];
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
}

void Spray::depositInto(float* dFoam, int n, cudaStream_t stream)
{
    const dim3 tpb(16, 16), bpg((n + 15) / 16, (n + 15) / 16);
    deposit_kernel<<<bpg, tpb, 0, stream>>>(dFoam, d_gridA, n);
}

void Spray::clearGrid(cudaStream_t stream)
{
    CK(cudaMemsetAsync(d_gridA, 0, (size_t)cfg_.n * cfg_.n * sizeof(int),
                       stream));
}

void Spray::finalize(cudaStream_t stream)
{
    const int tpb = 256, bpg = (maxP_ + tpb - 1) / tpb;
    spray_finalize_kernel<<<bpg, tpb, 0, stream>>>(d_parts, d_vel, d_age,
                                                   d_state, maxP_, d_render,
                                                   d_renderSize);
}

void Spray::release()
{
    cudaFree(d_emitList); cudaFree(d_emitCount); cudaFree(d_parts);
    cudaFree(d_vel); cudaFree(d_age); cudaFree(d_state);
    cudaFree(d_gridA); cudaFree(d_gridB); cudaFree(d_render);
    cudaFree(d_renderSize);
    d_emitList = d_emitCount = d_state = nullptr;
    d_gridA = d_gridB = nullptr;
    d_parts = d_vel = d_render = nullptr;
    d_renderSize = nullptr;
    d_age = nullptr;
}
