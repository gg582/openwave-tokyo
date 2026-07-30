// ============================================================================
// spray.h — ballistic spray droplet system (particle extension)
//
// Breaking lips eject droplets that follow ballistic trajectories and fall
// back into the sea — the sharp, rough spray a heightfield cannot show.
//
// A/B kernel structure (identical outputs, benchmarked):
//   emitter scan      traditional: one global atomicAdd per emitter cell
//                     complementary: warp-ballot + shuffle prefix scan, one
//                       atomicAdd per active warp (register exchange)
//   grid projection   traditional: naive per-particle atomicAdd
//                     complementary: warp-aggregated atomics (__match_any
//                       leader, one atomic per warp per cell)
//   ballistic update  shared kernel (deterministic hash RNG per particle)
//
// The emitter list is sorted on the host (tiny) and deposits are Q16.16
// fixed-point, so BOTH variants produce bit-identical particle states and
// bit-identical foam deposits — the videos stay binary-identical while the
// per-kernel timings expose the difference.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

struct SprayConfig {
    int   n           = 512;        // wave grid side
    int   maxEmitters = 4096;
    int   perEmitter  = 32;         // particles per emitter cell
    float breakThresh = 0.55f;      // air fraction to eject
    float domain      = 2500.0f;
    float phaseSpeed  = 9.5f;       // peak phase speed (m/s)
};

class Spray {
public:
    void init(const SprayConfig& cfg);
    void scan(const float* dFoam, int variant, cudaStream_t stream = 0);
    int  emitterCountHost();              // clamped count (D2H of 1 int)
    const int* emitterList() { return d_emitList; }
    void step(float tide, float dt, int frame, const float* dHeight,
              cudaStream_t stream = 0);
    void project(int variant, int gridSlot, cudaStream_t stream = 0);
    int* grid(int slot) { return slot ? d_gridB : d_gridA; }
    void depositInto(float* dFoam, int n, cudaStream_t stream = 0);
    void clearGrid(cudaStream_t stream = 0);
    void finalize(cudaStream_t stream = 0);
    void release();

    float4* positions() { return d_render; }
    float*  sizes()     { return d_renderSize; }
    int     particleCount() const { return maxP_; }
    int     maxParticles() const { return maxP_; }

    // bench accumulators (ms), per variant: 0 = traditional, 1 = complementary
    double msScan[2] = {0, 0}, msStep = 0, msProject[2] = {0, 0};
    int    framesScan[2] = {0, 0}, framesStep = 0, framesProject[2] = {0, 0};

private:
    SprayConfig cfg_;
    int*   d_emitList  = nullptr;
    int*   d_emitCount = nullptr;
    float4* d_parts    = nullptr;
    float4* d_vel      = nullptr;
    float*  d_age      = nullptr;
    int*    d_state    = nullptr;
    int*    d_gridA    = nullptr;
    int*    d_gridB    = nullptr;
    float4* d_render   = nullptr;
    float*  d_renderSize = nullptr;  // per-droplet diameter (m) for the VBO
    const float* d_heightField = nullptr;
    int     maxP_ = 0;
};
