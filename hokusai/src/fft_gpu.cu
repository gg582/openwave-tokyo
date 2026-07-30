// ============================================================================
// fft_gpu.cu — batch 1D FFT (complementary warp-shuffle vs traditional)
//
// Both implementations are decimation-in-time radix-2 with a bit-reversed
// input gather, so results are in natural order and bit-identical in
// structure; they differ ONLY in how the butterfly operands move:
//
//   complementary : lanes T_i / T_{i+16} form antipodal (complementary)
//                   pairs exchanging A and B*W through __shfl_xor_sync at
//                   register level (~1 cycle, no memory hierarchy). The
//                   add/subtract role is selected branch-free from the
//                   laneId bit via an IEEE-754 sign-bit XOR. Stages with
//                   span >= 32 cross warp boundaries and fall back to
//                   global-memory passes.
//   traditional   : every stage reads both operands from global memory and
//                   writes both results back (classic multi-pass
//                   Cooley-Tukey, the structure a generic library plan
//                   uses). This is the control group for the benchmark.
// ============================================================================
#include "fft_gpu.h"
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

__device__ __forceinline__ float2 cmulf(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// warp_butterfly — one radix-2 stage entirely in registers.
// mask = span (1..16); lane T_i pairs with antipodal lane T_{i^mask}.
// Primary ((lane&mask)==0) computes A + B*W, secondary computes A - B*W.
// B*W travels between the pair through __shfl_xor_sync — no shared memory,
// no bank conflicts. Add/subtract is a branch-free sign-bit XOR.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 warp_butterfly(float2 v, float2 w,
                                                 unsigned mask, int lane,
                                                 float dirSign)
{
    const float2 p = cmulf(w, v);                        // B*W (warp-wide)

    const unsigned sec  = (unsigned)(lane & mask);       // role bit
    const float2   send = sec ? p : v;                   // SEL: ship A or B*W

    float2 r;                                            // register exchange
    r.x = __shfl_xor_sync(0xffffffffu, send.x, mask);
    r.y = __shfl_xor_sync(0xffffffffu, send.y, mask);

    // Role bit -> IEEE-754 sign flip (bit 31): pure bit masking, no branch.
    const int flip = (int)(sec << __clz(mask));

    const float2 a = sec ? r : v;                        // SEL: A operand
    const float2 t = sec ? p : r;                        // SEL: B*W operand

    float2 out;
    out.x = a.x + __int_as_float(__float_as_int(t.x) ^ flip);
    out.y = a.y + __int_as_float(__float_as_int(t.y) ^ flip);
    (void)dirSign;
    return out;
}

// ---------------------------------------------------------------------------
// Complementary kernel: batched rows, register-resident stages (spans 1..16)
// Each warp owns 32 consecutive samples of one row. The transform-local
// index is bit-reversed over logN bits; row offset passes through.
// inverse=true flips the twiddle sign (IFFT); caller applies the 1/n scale.
// ---------------------------------------------------------------------------
__global__ void complementary_warp_kernel(const float2* __restrict__ in,
                                          float2* __restrict__ out,
                                          int n, int logN, int inverse,
                                          int totalChunks)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= totalChunks) return;

    // Flat warp id -> (row, 32-sample chunk within the row).
    const int chunksRow = n >> 5;
    const int row       = warp / chunksRow;
    const int chunk     = warp - row * chunksRow;
    const int base      = row * n + (chunk << 5);

    // Bit-reversed gather over the transform-local index (logN bits); the
    // row offset (bits above logN) passes through unchanged.
    const unsigned g   = (unsigned)(base + lane);
    const unsigned rev = (g & ~((1u << logN) - 1u)) | (__brev(g) >> (32 - logN));
    float2 v = in[rev];

    const float sgn = inverse ? 1.0f : -1.0f;
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        const unsigned mask = 1u << s;
        const int   k     = lane & (int)(mask - 1u);
        const float angle = sgn * 2.0f * CUDART_PI_F * (float)k / (float)(2u * mask);
        float2 w;
        sincosf(angle, &w.y, &w.x);
        v = warp_butterfly(v, w, mask, lane, sgn);
    }

    out[base + lane] = v;
}

// ---------------------------------------------------------------------------
// Cross-warp butterfly stage (span >= 32), batched. One thread per
// butterfly. Shared by both pipelines for the high stages; the complementary
// path simply needs 5 fewer of them (the register-resident ones).
// ---------------------------------------------------------------------------
__global__ void stage_kernel(float2* __restrict__ data, int n, int halfRow,
                             int s, int inverse, int totalHalf)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalHalf) return;

    // Flat butterfly id -> (row, butterfly within the row).
    const int row   = tid / halfRow;
    const int local = tid - row * halfRow;

    const unsigned mask = 1u << s;
    const int k = local & (int)(mask - 1u);
    const int iLocal = ((local >> s) << (s + 1)) | k;
    const int i = row * n + iLocal;
    const int j = i + (int)mask;

    const float sgn = inverse ? 1.0f : -1.0f;
    const float angle = sgn * 2.0f * CUDART_PI_F * (float)k / (float)(2u * mask);
    float2 w;
    sincosf(angle, &w.y, &w.x);

    const float2 a = data[i];
    const float2 b = data[j];
    const float2 t = cmulf(w, b);
    data[i] = make_float2(a.x + t.x, a.y + t.y);
    data[j] = make_float2(a.x - t.x, a.y - t.y);
}

// Guard-carrying variant launch bounds helper: we simply compute total
// threads on the host and never overshoot (grid-stride not needed).

// ---------------------------------------------------------------------------
// Traditional pipeline: bit-reversed permutation + one global pass per stage
// (including the tiny spans a register/shuffle kernel handles for free).
// ---------------------------------------------------------------------------
__global__ void bitrev_copy_kernel(const float2* __restrict__ in,
                                   float2* __restrict__ out, int n, int logN,
                                   int total)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const unsigned rev = ((unsigned)e & ~((1u << logN) - 1u))
                       | (__brev((unsigned)e) >> (32 - logN));
    out[e] = in[rev];
}

__global__ void scale_kernel(float2* data, float inv, int total)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    data[e].x *= inv;
    data[e].y *= inv;
}

// ---------------------------------------------------------------------------
// 32x32 tiled transpose (shared memory, both pipelines).
// ---------------------------------------------------------------------------
__global__ void transpose_kernel(const float2* __restrict__ in,
                                 float2* __restrict__ out, int n)
{
    __shared__ float2 tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
static inline int ilog2(int n) { int l = 0; while ((1 << l) < n) ++l; return l; }

// cached scratch buffers (avoid per-call cudaMalloc overhead in hot loops)
// slot 0: 2D transpose; slot 1: traditional-stage work buffer — kept separate
// because fft_2d uses both at once
static float2* g_scratch[2] = { nullptr, nullptr };
static size_t  g_scratchBytes[2] = { 0, 0 };
static float2* scratchFor(int slot, size_t bytes)
{
    if (bytes > g_scratchBytes[slot]) {
        if (g_scratch[slot]) cudaFree(g_scratch[slot]);
        CK(cudaMalloc(&g_scratch[slot], bytes));
        g_scratchBytes[slot] = bytes;
    }
    return g_scratch[slot];
}

void batch_fft_1d(float2* data, int n, int batch, bool inverse,
                  FftMode mode, cudaStream_t stream)
{
    const int logN  = ilog2(n);
    const int total = n * batch;
    const int inv   = inverse ? 1 : 0;

    if (mode == FFT_COMPLEMENTARY) {
        // The warp kernel gathers its operands in bit-reversed order from
        // the WHOLE row, so it must not run in place: warps would overwrite
        // locations other warps still need to read. Gather from a scratch
        // copy instead (identical values, race-free), then butterflies are
        // pairwise and safe to update in place.
        float2* scratch = scratchFor(1, (size_t)total * sizeof(float2));
        CK(cudaMemcpyAsync(scratch, data, (size_t)total * sizeof(float2),
                           cudaMemcpyDeviceToDevice, stream));
        // One warp per 32-sample chunk of every row.
        const int totalChunks = (n >> 5) * batch;
        const int warpsPerBlock = 8;
        const int blocks = (totalChunks + warpsPerBlock - 1) / warpsPerBlock;
        complementary_warp_kernel<<<blocks, warpsPerBlock * 32, 0, stream>>>(
            scratch, data, n, logN, inv, totalChunks);
        // Remaining stages (span >= 32) as global butterfly passes.
        const int halfRow = n >> 1;
        const int totalHalf = halfRow * batch;
        const int tpb = 256;
        const int bpg = (totalHalf + tpb - 1) / tpb;
        for (int s = 5; s < logN; ++s)
            stage_kernel<<<bpg, tpb, 0, stream>>>(data, n, halfRow, s, inv,
                                                  totalHalf);
    } else {
        // Traditional: bit-reversed copy + logN full global passes.
        float2* scratch = scratchFor(1, (size_t)total * sizeof(float2));
        const int tpb = 256;
        const int bpg = (total + tpb - 1) / tpb;
        bitrev_copy_kernel<<<bpg, tpb, 0, stream>>>(data, scratch, n, logN, total);
        const int halfRow = n >> 1;
        const int totalHalf = halfRow * batch;
        const int bph = (totalHalf + tpb - 1) / tpb;
        for (int s = 0; s < logN; ++s)   // ALL stages through global memory
            stage_kernel<<<bph, tpb, 0, stream>>>(scratch, n, halfRow, s, inv,
                                                  totalHalf);
        CK(cudaMemcpyAsync(data, scratch, (size_t)total * sizeof(float2),
                           cudaMemcpyDeviceToDevice, stream));
    }

    if (inverse) {
        const int tpb = 256;
        const int bpg = (total + tpb - 1) / tpb;
        scale_kernel<<<bpg, tpb, 0, stream>>>(data, 1.0f / (float)n, total);
    }
}

void fft_2d(float2* data, int n, bool inverse, FftMode mode,
            cudaStream_t stream)
{
    float2* tmp = scratchFor(0, (size_t)n * n * sizeof(float2));
    const dim3 tpb(32, 32);
    const dim3 bpg((n + 31) / 32, (n + 31) / 32);

    batch_fft_1d(data, n, n, inverse, mode, stream);               // rows
    transpose_kernel<<<bpg, tpb, 0, stream>>>(data, tmp, n);
    batch_fft_1d(tmp, n, n, inverse, mode, stream);                // columns
    transpose_kernel<<<bpg, tpb, 0, stream>>>(tmp, data, n);
}

// ---------------------------------------------------------------------------
// Micro-benchmark
// ---------------------------------------------------------------------------
void fft_bench(int n, int iters, float* msTraditional, float* msComplementary)
{
    float2* d = nullptr;
    CK(cudaMalloc(&d, (size_t)n * n * sizeof(float2)));
    CK(cudaMemset(d, 0, (size_t)n * n * sizeof(float2)));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    for (int m = 0; m < 2; ++m) {
        const FftMode mode = m ? FFT_COMPLEMENTARY : FFT_TRADITIONAL;
        fft_2d(d, n, true, mode);                                  // warm-up
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) fft_2d(d, n, true, mode);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        if (m) *msComplementary = ms / iters;
        else   *msTraditional   = ms / iters;
    }
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    CK(cudaFree(d));
}
