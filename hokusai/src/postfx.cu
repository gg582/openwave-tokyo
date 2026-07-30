// ============================================================================
// postfx.cu — warp-shuffle inverse-OTF unsharp masking + RGBA->NV12
// ============================================================================
#include "postfx.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(call)                                                            \
    do {                                                                    \
        cudaError_t e_ = (call);                                            \
        if (e_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(e_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__device__ __forceinline__ float3 u2f(uchar4 c)
{
    return make_float3((float)c.x, (float)c.y, (float)c.z);
}
__device__ __forceinline__ uchar4 f2u(float3 v)
{
    return make_uchar4((unsigned char)fminf(fmaxf(v.x, 0.f), 255.f),
                       (unsigned char)fminf(fmaxf(v.y, 0.f), 255.f),
                       (unsigned char)fminf(fmaxf(v.z, 0.f), 255.f), 255);
}

// bilinear fetch of one channel from the uchar4 frame (for chroma re-align)
__device__ __forceinline__ float sampleChannel(const uchar4* img, int w, int h,
                                               float x, float y, int ch)
{
    x = fminf(fmaxf(x, 0.f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.f), (float)(h - 1));
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    auto chOf = [&](int xx, int yy) {
        const uchar4 c = img[yy * w + xx];
        return (float)(ch == 0 ? c.x : ch == 1 ? c.y : c.z);
    };
    const float a = chOf(x0, y0) * (1 - fx) + chOf(x1, y0) * fx;
    const float b = chOf(x0, y1) * (1 - fx) + chOf(x1, y1) * fx;
    return a * (1 - fy) + b * fy;
}

// ---------------------------------------------------------------------------
// Shared per-pixel chroma re-alignment (identical arithmetic on both paths)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 alignedPixel(const uchar4* __restrict__ in,
                                               int w, int h, int x, int y,
                                               float chromaK)
{
    const float cx = (float)x - 0.5f * (float)w;
    const float cy = (float)y - 0.5f * (float)h;
    const float r2 = (cx * cx + cy * cy) / (0.25f * (float)(w * w));
    const float dx = chromaK * r2 * cx;
    const float dy = chromaK * r2 * cy;
    float3 c;
    c.x = sampleChannel(in, w, h, (float)x - dx, (float)y - dy, 0);
    c.y = (float)in[y * w + x].y;
    c.z = sampleChannel(in, w, h, (float)x + dx, (float)y + dy, 2);
    return c;
}

// ---------------------------------------------------------------------------
// Pass 1 (horizontal), COMPLEMENTARY: chroma re-alignment + 5-tap Gaussian
// [1 4 6 4 1]/16. Neighbor pixels at +/-1 and +/-2 arrive through
// __shfl_up/down_sync — antipodal lane-pair register exchange (~1 cycle, no
// shared memory, no extra global traffic). Edge taps use the SAME
// alignedPixel() evaluation as the traditional variant so results are
// bit-identical.
// ---------------------------------------------------------------------------
__global__ void correct_h_kernel(const uchar4* __restrict__ in,
                                 float3* __restrict__ tmp,
                                 int w, int h, float chromaK)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y;
    if (gx >= w || gy >= h) return;
    const int lane = threadIdx.x & 31;

    float3 c = alignedPixel(in, w, h, gx, gy, chromaK);

    float3 l1, l2, rt1, rt2;
    l1.x = __shfl_up_sync(0xffffffffu, c.x, 1);
    l1.y = __shfl_up_sync(0xffffffffu, c.y, 1);
    l1.z = __shfl_up_sync(0xffffffffu, c.z, 1);
    rt1.x = __shfl_down_sync(0xffffffffu, c.x, 1);
    rt1.y = __shfl_down_sync(0xffffffffu, c.y, 1);
    rt1.z = __shfl_down_sync(0xffffffffu, c.z, 1);
    l2.x = __shfl_up_sync(0xffffffffu, c.x, 2);
    l2.y = __shfl_up_sync(0xffffffffu, c.y, 2);
    l2.z = __shfl_up_sync(0xffffffffu, c.z, 2);
    rt2.x = __shfl_down_sync(0xffffffffu, c.x, 2);
    rt2.y = __shfl_down_sync(0xffffffffu, c.y, 2);
    rt2.z = __shfl_down_sync(0xffffffffu, c.z, 2);

    // Off-warp / off-image taps (shfl out-of-range would silently return
    // the calling lane's own value): evaluate those taps from global memory
    // with the SAME alignedPixel() arithmetic as the traditional variant.
    if (lane == 0 || gx == 0) {
        l1 = alignedPixel(in, w, h, max(gx - 1, 0), gy, chromaK);
        l2 = alignedPixel(in, w, h, max(gx - 2, 0), gy, chromaK);
    }
    if (lane == 1 || gx == 1)
        l2 = alignedPixel(in, w, h, max(gx - 2, 0), gy, chromaK);
    if (lane == 31 || gx == w - 1) {
        rt1 = alignedPixel(in, w, h, min(gx + 1, w - 1), gy, chromaK);
        rt2 = alignedPixel(in, w, h, min(gx + 2, w - 1), gy, chromaK);
    }
    if (lane == 30 || gx == w - 2)
        rt2 = alignedPixel(in, w, h, min(gx + 2, w - 1), gy, chromaK);

    float3 blur;
    blur.x = (l2.x + 4.f * l1.x + 6.f * c.x + 4.f * rt1.x + rt2.x) * (1.f / 16.f);
    blur.y = (l2.y + 4.f * l1.y + 6.f * c.y + 4.f * rt1.y + rt2.y) * (1.f / 16.f);
    blur.z = (l2.z + 4.f * l1.z + 6.f * c.z + 4.f * rt1.z + rt2.z) * (1.f / 16.f);
    tmp[gy * w + gx] = blur;
}

// ---------------------------------------------------------------------------
// Pass 1 (horizontal), TRADITIONAL control: the SAME algorithm, but every
// tap is a plain global-memory fetch — no register exchange anywhere.
// Operation-for-operation identical to the shuffle variant, so the outputs
// must compare bit-equal.
// ---------------------------------------------------------------------------
__global__ void correct_h_trad_kernel(const uchar4* __restrict__ in,
                                      float3* __restrict__ tmp,
                                      int w, int h, float chromaK)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y;
    if (gx >= w || gy >= h) return;

    const int xm2 = max(gx - 2, 0), xm1 = max(gx - 1, 0);
    const int xp1 = min(gx + 1, w - 1), xp2 = min(gx + 2, w - 1);
    const float3 l2  = alignedPixel(in, w, h, xm2, gy, chromaK);
    const float3 l1  = alignedPixel(in, w, h, xm1, gy, chromaK);
    const float3 c   = alignedPixel(in, w, h, gx,  gy, chromaK);
    const float3 rt1 = alignedPixel(in, w, h, xp1, gy, chromaK);
    const float3 rt2 = alignedPixel(in, w, h, xp2, gy, chromaK);

    float3 blur;
    blur.x = (l2.x + 4.f * l1.x + 6.f * c.x + 4.f * rt1.x + rt2.x) * (1.f / 16.f);
    blur.y = (l2.y + 4.f * l1.y + 6.f * c.y + 4.f * rt1.y + rt2.y) * (1.f / 16.f);
    blur.z = (l2.z + 4.f * l1.z + 6.f * c.z + 4.f * rt1.z + rt2.z) * (1.f / 16.f);
    tmp[gy * w + gx] = blur;
}

// ---------------------------------------------------------------------------
// Pass 2 (vertical): 5-tap Gaussian from the intermediate, then the
// inverse-OTF unsharp combine with field-dependent gain:
//   g(r) = amount * (1 + alpha * r^4)
// The aberration blur grows ~ r^4, so the correction grows the same way —
// center stays untouched, peripheral edge contrast is restored.
// ---------------------------------------------------------------------------
__global__ void correct_v_kernel(const float3* __restrict__ tmp,
                                 const uchar4* __restrict__ orig,
                                 uchar4* __restrict__ out,
                                 int w, int h, float amount, float alpha)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= w || gy >= h) return;
    const int e = gy * w + gx;

    const int ym2 = max(gy - 2, 0), ym1 = max(gy - 1, 0);
    const int yp1 = min(gy + 1, h - 1), yp2 = min(gy + 2, h - 1);
    const float3 tm2 = tmp[ym2 * w + gx];
    const float3 tm1 = tmp[ym1 * w + gx];
    const float3 tc  = tmp[e];
    const float3 tp1 = tmp[yp1 * w + gx];
    const float3 tp2 = tmp[yp2 * w + gx];

    float3 blur;
    blur.x = (tm2.x + 4.f * tm1.x + 6.f * tc.x + 4.f * tp1.x + tp2.x) * (1.f / 16.f);
    blur.y = (tm2.y + 4.f * tm1.y + 6.f * tc.y + 4.f * tp1.y + tp2.y) * (1.f / 16.f);
    blur.z = (tm2.z + 4.f * tm1.z + 6.f * tc.z + 4.f * tp1.z + tp2.z) * (1.f / 16.f);

    const float3 o = u2f(orig[e]);
    const float cx = (float)gx - 0.5f * (float)w;
    const float cy = (float)gy - 0.5f * (float)h;
    const float r2 = (cx * cx + cy * cy) / (0.25f * (float)(w * w));
    const float gain = amount * (1.0f + alpha * r2 * r2);

    float3 res;
    res.x = o.x + gain * (o.x - blur.x);
    res.y = o.y + gain * (o.y - blur.y);
    res.z = o.z + gain * (o.z - blur.z);
    out[e] = f2u(res);
}

// ---------------------------------------------------------------------------
// RGBA -> NV12 (BT.709, limited range), 2x2 chroma subsampling
// ---------------------------------------------------------------------------
__global__ void rgba_to_nv12_kernel(const uchar4* __restrict__ rgba, int w, int h,
                                    unsigned char* __restrict__ y, int yPitch,
                                    unsigned char* __restrict__ uv, int uvPitch)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= w || gy >= h) return;

    // glReadPixels delivers bottom-up rows: flip here so the MP4 is upright
    const uchar4 c = rgba[(h - 1 - gy) * w + gx];
    const float R = (float)c.x, G = (float)c.y, B = (float)c.z;
    const float Y = 16.0f + 0.2126f * R + 0.7152f * G + 0.0722f * B;
    y[gy * yPitch + gx] = (unsigned char)fminf(fmaxf(Y, 0.f), 255.f);

    if ((gx & 1) == 0 && (gy & 1) == 0) {
        // average the 2x2 block for chroma
        float r = R, g = G, b = B;
        const int x1 = min(gx + 1, w - 1), y1 = max(gy - 1, 0);
        const uchar4 c10 = rgba[(h - 1 - gy) * w + x1];
        const uchar4 c01 = rgba[(h - 1 - y1) * w + gx];
        const uchar4 c11 = rgba[(h - 1 - y1) * w + x1];
        r += (float)c10.x + (float)c01.x + (float)c11.x;
        g += (float)c10.y + (float)c01.y + (float)c11.y;
        b += (float)c10.z + (float)c01.z + (float)c11.z;
        r *= 0.25f; g *= 0.25f; b *= 0.25f;
        const float U = 128.0f - 0.1146f * r - 0.3854f * g + 0.5f * b;
        const float V = 128.0f + 0.5f * r - 0.4542f * g - 0.0458f * b;
        const int cu = (gy >> 1) * uvPitch + (gx & ~1);
        uv[cu + 0] = (unsigned char)fminf(fmaxf(U, 0.f), 255.f);
        uv[cu + 1] = (unsigned char)fminf(fmaxf(V, 0.f), 255.f);
    }
}

void rgbaToNv12(const uchar4* rgba, int w, int h,
                unsigned char* y, int yPitch,
                unsigned char* uv, int uvPitch, cudaStream_t stream)
{
    const dim3 tpb(16, 16), bpg((w + 15) / 16, (h + 15) / 16);
    rgba_to_nv12_kernel<<<bpg, tpb, 0, stream>>>(rgba, w, h, y, yPitch,
                                                 uv, uvPitch);
}

// ---------------------------------------------------------------------------
// PostFx class
// ---------------------------------------------------------------------------
bool PostFx::init(int w, int h, unsigned pbo)
{
    w_ = w; h_ = h; pbo_ = pbo;
    CK(cudaMalloc(&d_orig_, (size_t)w * h * sizeof(uchar4)));
    CK(cudaMalloc(&d_tmp_,  (size_t)w * h * sizeof(float3)));
    const cudaError_t e = cudaGraphicsGLRegisterBuffer(&resPbo_, pbo,
                                                       cudaGraphicsMapFlagsNone);
    if (e != cudaSuccess) {
        fprintf(stderr, "PBO interop registration failed: %s\n",
                cudaGetErrorString(e));
        return false;
    }
    registered_ = true;
    return true;
}

uchar4* PostFx::beginFrame(int correctionMode, float amount,
                           cudaStream_t stream)
{
    // correctionMode: 1 = complementary (warp-shuffle taps),
    //                 0 = traditional control (global-memory taps),
    //                -1 = none
    CK(cudaGraphicsMapResources(1, &resPbo_, stream));
    size_t sz = 0;
    uchar4* frame = nullptr;
    CK(cudaGraphicsResourceGetMappedPointer((void**)&frame, &sz, resPbo_));

    if (correctionMode >= 0) {
        lastCorrectionMs = 0.0f;
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0, stream));
        // chroma re-align coefficient mirrors the aberration shader's k*r^2
        const float chromaK = amount * 0.0022f * 4.0f;
        {
            const int tpb = 256;
            const dim3 bpg((w_ + tpb - 1) / tpb, h_);
            if (correctionMode == 1)
                correct_h_kernel<<<bpg, tpb, 0, stream>>>(d_orig_, d_tmp_,
                                                          w_, h_, chromaK);
            else
                correct_h_trad_kernel<<<bpg, tpb, 0, stream>>>(d_orig_, d_tmp_,
                                                               w_, h_, chromaK);
        }
        {
            const dim3 tpb(16, 16), bpg((w_ + 15) / 16, (h_ + 15) / 16);
            correct_v_kernel<<<bpg, tpb, 0, stream>>>(d_tmp_, d_orig_, frame,
                                                      w_, h_, 1.2f, 0.9f);
        }
        CK(cudaEventRecord(t1, stream));
        CK(cudaEventSynchronize(t1));
        CK(cudaEventElapsedTime(&lastCorrectionMs, t0, t1));
        CK(cudaEventDestroy(t0));
        CK(cudaEventDestroy(t1));
    } else {
        lastCorrectionMs = 0.0f;
    }
    return frame;
}

void PostFx::snapshotFrame(cudaStream_t stream)
{
    CK(cudaGraphicsMapResources(1, &resPbo_, stream));
    size_t sz = 0;
    uchar4* frame = nullptr;
    CK(cudaGraphicsResourceGetMappedPointer((void**)&frame, &sz, resPbo_));
    // pristine copy of this frame's render for all pipelines to read from
    CK(cudaMemcpyAsync(d_orig_, frame, (size_t)w_ * h_ * sizeof(uchar4),
                       cudaMemcpyDeviceToDevice, stream));
    CK(cudaGraphicsUnmapResources(1, &resPbo_, stream));
}

void PostFx::endFrame(cudaStream_t stream)
{
    CK(cudaGraphicsUnmapResources(1, &resPbo_, stream));
}

void PostFx::release()
{
    if (registered_) cudaGraphicsUnregisterResource(resPbo_);
    cudaFree(d_orig_);
    cudaFree(d_tmp_);
    resPbo_ = nullptr;
    d_orig_ = nullptr;
    d_tmp_ = nullptr;
}
