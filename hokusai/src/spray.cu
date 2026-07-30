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
__global__ void scan_atomic_kernel(const float* __restrict__ foam, int n,
                                   float thr, int* __restrict__ list,
                                   int* __restrict__ counter)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    if (foam[j * n + i] > thr) {
        const int slot = atomicAdd(counter, 1);
        list[slot] = j * n + i;
    }
}

// ---------------------------------------------------------------------------
// emitter scan — COMPLEMENTARY: warp-ballot + shuffle prefix scan.
// The 32 lanes of a warp test 32 cells; the ballot mask is prefix-scanned
// in registers (antipodal lane exchange) and only the warp leader touches
// the global counter — one atomicAdd per active warp instead of per cell.
// ---------------------------------------------------------------------------
__global__ void scan_shuffle_kernel(const float* __restrict__ foam, int n,
                                    float thr, int* __restrict__ list,
                                    int* __restrict__ counter)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x & 31;

    bool pred = false;
    int cell = 0;
    if (i < n && j < n) {
        cell = j * n + i;
        pred = foam[cell] > thr;
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

void sprayScanAtomic(const float* foam, int n, float thr, int* list,
                     int* counter, cudaStream_t s)
{
    const dim3 tpb(32, 8), bpg((n + 31) / 32, (n + 7) / 8);
    scan_atomic_kernel<<<bpg, tpb, 0, s>>>(foam, n, thr, list, counter);
}
void sprayScanShuffle(const float* foam, int n, float thr, int* list,
                      int* counter, cudaStream_t s)
{
    const dim3 tpb(32, 8), bpg((n + 31) / 32, (n + 7) / 8);
    scan_shuffle_kernel<<<bpg, tpb, 0, s>>>(foam, n, thr, list, counter);
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

            if (r1 < 0.55f) {
                // Most respawns become INDIVIDUAL FOAM BUBBLES directly on
                // the surface: each one is a discrete air-entrainment bubble
                // rafted at the breaking crest, with its own diameter
                // (3 cm .. 14 cm, heavily skewed small like real sea foam).
                const float surf = heightField[cell] + tide;
                const float diam = 0.03f + 0.11f * r5 * r5;
                parts[i] = make_float4(wx + (r2 - 0.5f) * 6.0f, surf + 0.02f,
                                       wz + (r4 - 0.5f) * 6.0f, 0.0f);
                vel[i] = make_float4(0.0f, 0.0f, 0.0f, diam);
                age[i] = 0.0f;
                state[i] = 2;
            } else {
                // ballistic ejection: forward at ~phase speed, up like the lip
                const float v0 = phaseSpeed * (0.8f + 0.5f * r1);
                vel[i] = make_float4(pdir.x * v0 + (r2 - 0.5f) * 2.0f,
                                     0.4f + 1.6f * r3 * phaseSpeed * 0.35f,
                                     pdir.z * v0 + (r4 - 0.5f) * 2.0f,
                                     0.02f + 0.05f * r5);   // droplet diameter
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
        const float diam = vel[i].w;
        float3 v = make_float3(vel[i].x, vel[i].y, vel[i].z);
        v.y -= 9.81f * dt;
        float3 p = make_float3(parts[i].x + v.x * dt,
                               parts[i].y + v.y * dt,
                               parts[i].z + v.z * dt);
        vel[i] = make_float4(v.x, v.y, v.z, diam);
        parts[i] = make_float4(p.x, p.y, p.z, 0.0f);
        age[i] += dt;

        // surface height at the droplet position
        const int gx = min(max((int)((p.x / domain + 0.5f) * n), 0), n - 1);
        const int gy = min(max((int)((p.z / domain + 0.5f) * n), 0), n - 1);
        const float surf = heightField[gy * n + gx] + tide;
        if (p.y <= surf || age[i] > 2.5f) {
            parts[i].y = surf + 0.03f;
            // a landed spray droplet merges into the foam: it becomes an
            // individual floating bubble of its own size
            vel[i].w = fmaxf(diam * (1.5f + shash((unsigned)i * 747796405u +
                                                  (unsigned)frame)), 0.05f);
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
        const float surf = heightField[gy * n + gx] + tide;
        // drift along propagation like rafted foam does
        const float drift = 0.35f * dt;
        parts[i] = make_float4(p.x + pdir.x * drift, surf + 0.03f,
                               p.z + pdir.z * drift, 0.0f);
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
    if (state[i] == 1) alpha = fminf(age[i] * 5.0f, 1.0f);      // flying
    else if (state[i] == 2) {                                    // floating
        const unsigned s1 = (unsigned)i * 2654435761u;
        const float lifetime = 1.2f + 2.0f * shash(s1);
        alpha = 0.85f * fminf(fmaxf(1.0f - age[i] / lifetime, 0.0f) + 0.15f,
                              1.0f);
    } else if (state[i] == 3) {                                  // just landed
        alpha = 1.0f;
        state[i] = 2;
    }
    render[i] = make_float4(parts[i].x, parts[i].y, parts[i].z, alpha);
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

void Spray::scan(const float* dFoam, int variant, cudaStream_t stream)
{
    const int n = cfg_.n;
    CK(cudaMemsetAsync(d_emitCount, 0, sizeof(int), stream));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, stream));
    if (variant == 1)
        sprayScanShuffle(dFoam, n, cfg_.breakThresh, d_emitList, d_emitCount,
                         stream);
    else
        sprayScanAtomic(dFoam, n, cfg_.breakThresh, d_emitList, d_emitCount,
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
