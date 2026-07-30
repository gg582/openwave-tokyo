// ============================================================================
// postfx.h — CUDA post-processing: warp-shuffle inverse-OTF unsharp kernel
//
// The complementary pipeline sharpens the aberration pass output with a
// CUDA kernel that:
//   1. re-aligns the lateral chromatic aberration (R/B channels sampled at
//      the inverse radial displacement) — the inverse of the color fringe
//      term of the simulated optical transfer function
//   2. applies a 5-tap Gaussian unsharp mask whose horizontal neighbor
//      exchange runs ENTIRELY in registers through __shfl_up/down_sync
//      (antipodal lane-pair exchange, ~1 cycle, zero shared/global traffic),
//      with a field-dependent gain g(r) = amount * (1 + alpha * r^4)
//      approximating the inverse OTF of the r^4 spherical blur
//
// Zero-copy frame flow: GL PBO -> cudaGraphics mapped pointer -> correction
// kernel (in place) -> RGBA->NV12 kernel writing directly into the FFmpeg
// CUDA hw_frames buffers consumed by NVENC.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class PostFx {
public:
    bool init(int w, int h, unsigned pbo);
    // Map the PBO and run the correction kernel.
    // correctionMode: 1 = complementary (warp-shuffle register taps),
    //                 0 = traditional control (global-memory taps),
    //                -1 = none. Returns the device RGBA frame pointer
    // (valid until endFrame()).
    uchar4* beginFrame(int correctionMode, float amount, cudaStream_t stream);
    // Copy the pristine rendered frame into the internal source buffer;
    // call once per frame BEFORE any pipeline's beginFrame, so every
    // pipeline corrects the same pristine input (not another pipeline's
    // output).
    void snapshotFrame(cudaStream_t stream);
    void endFrame(cudaStream_t stream);
    void release();

    float lastCorrectionMs = 0.0f;   // timing of the correction kernel

private:
    int w_ = 0, h_ = 0;
    unsigned pbo_ = 0;
    cudaGraphicsResource* resPbo_ = nullptr;
    uchar4* d_orig_ = nullptr;   // pristine copy of the aberration output
    float3* d_tmp_  = nullptr;   // horizontal blur intermediate
    bool registered_ = false;
};

// RGBA (uchar4) -> NV12 planes (BT.709 limited range), writing straight
// into encoder-owned device buffers.
void rgbaToNv12(const uchar4* rgba, int w, int h,
                unsigned char* y, int yPitch,
                unsigned char* uv, int uvPitch, cudaStream_t stream);
