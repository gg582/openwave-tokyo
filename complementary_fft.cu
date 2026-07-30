// ============================================================================
// complementary_fft.cu
//
// 1D Radix-2 FFT butterfly kernel exploiting the antipodal (complementary
// pair) symmetry of the twiddle-factor structure:
//
//   Inside a 32-thread warp, lane T_i and lane T_{i+16} form a complementary
//   pair. For every butterfly stage with span `mask`:
//     - Primary   lanes ((lane & mask) == 0) compute  A + B*W   (sum branch)
//     - Secondary lanes ((lane & mask) != 0) compute  A - B*W   (diff branch)
//
//   The operand B*W is exchanged between the two lanes of a pair at the
//   REGISTER level with __shfl_xor_sync() — one instruction, no shared
//   memory, no bank conflicts, no __syncwarp().
//
//   Add/Subtract selection is done branch-free: the role bit is extracted
//   from laneId with pure bit-masking and applied as an XOR on the IEEE-754
//   sign bit, so the warp never diverges (no if/else anywhere).
//
// Two operating modes:
//   1) Chunk mode (default): each warp computes one independent 32-point
//      FFT — the five stages (spans 1..16) run entirely in registers.
//   2) Full-FFT mode: one N-point FFT with N = 2^L (L >= 5). The warp
//      kernel performs the first 5 stages (bit-reversed load over the full
//      N index, then spans 1..16 in registers); the remaining L-5 stages
//      pair lanes across warp boundaries and are executed as global-memory
//      butterfly passes (butterfly_stage_kernel).
//
// ----------------------------------------------------------------------------
// Bandwidth-overhead principle (vs. a generic cuFFT plan):
//
//   A general-purpose cuFFT plan for arbitrary N must execute each radix
//   stage as a separate pass over global memory (intermediate results are
//   spilled to DRAM between stages). An N-point transform then costs up to
//   log2(N) read+write round trips: traffic ~= 2 * log2(N) * N * 8 B.
//
//   Here the first 5 stages (the entire 32-point granularity) live in
//   registers. The only data movement between those stages is
//   __shfl_xor_sync(), which travels through the register file / shuffle
//   unit (~1 cycle, zero memory-hierarchy traffic). DRAM passes drop from
//   log2(N) to log2(N) - 4: for N = 2^16 that is 11 passes instead of 16
//   (~31% less traffic), and in chunk mode (32-point granularity) each
//   element touches DRAM exactly twice — one load, one store — a 5x
//   reduction versus staging a 32-point FFT through memory. (cuFFT's own
//   small-N specialized kernels use the same trick; the win here is against
//   generic multi-pass plans and shared-memory staged implementations.)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Complex multiply: c = a * b
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// warp_butterfly: one Radix-2 butterfly stage executed entirely in registers.
//
//   v    : this lane's operand (its own complex sample)
//   w    : twiddle factor for this lane/stage, W = exp(-2*pi*i*k/(2*mask))
//   mask : butterfly span for this stage (1, 2, 4, 8, 16). Lane T_i is
//          paired with antipodal lane T_{i ^ mask}; for the top stage
//          (mask = 16) this is exactly the T_i / T_{i+16} complementary pair.
//   lane : laneId (threadIdx.x & 31)
//
// Data exchange:
//   Primary lane sends A (raw operand), secondary lane sends B*W (twiddled
//   product). __shfl_xor_sync() performs the swap in both directions with a
//   single instruction — no shared memory, no bank conflicts.
//
// Divergence control:
//   The primary/secondary role bit is isolated with (lane & mask) and turned
//   into an IEEE-754 sign-bit flip via `sec << __clz(mask)` — pure bit
//   masking, no comparison, no branch. Add vs. subtract becomes
//   `a + (t XOR sign_flip)`, resolved inline at the (virtual) register level.
//   The remaining ternaries lower to predicated SEL instructions, so the
//   warp executes with zero divergence.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 warp_butterfly(float2 v, float2 w,
                                                 unsigned mask, int lane)
{
    // B*W: computed by every lane. Only the secondary lane's product is
    // consumed, but computing it warp-wide costs one FMA cluster and keeps
    // the code path uniform (no divergence).
    const float2 p = cmul(w, v);

    // Role bit from laneId: 0 -> primary (A + B*W), nonzero -> secondary.
    const unsigned sec = (unsigned)(lane & mask);

    // Select the outgoing register: primary ships A, secondary ships B*W.
    const float2 send = sec ? p : v;                      // SEL, not a branch

    // Register-level antipodal exchange: T_i <-> T_{i+mask}, ~1 cycle.
    float2 r;
    r.x = __shfl_xor_sync(0xffffffffu, send.x, mask);
    r.y = __shfl_xor_sync(0xffffffffu, send.y, mask);

    // Branchless add/subtract: turn the role bit into a sign-bit mask.
    // mask is a power of two, so (lane & mask) holds a single set bit at
    // position log2(mask); shifting left by __clz(mask) moves it to bit 31.
    const int flip = (int)(sec << __clz(mask));           // 0x00000000 or 0x80000000

    // A operand: primary keeps its own v, secondary takes the received A.
    const float2 a = sec ? r : v;                         // SEL
    // B*W term: primary takes the received product, secondary its own.
    const float2 t = sec ? p : r;                         // SEL

    float2 out;
    out.x = a.x + __int_as_float(__float_as_int(t.x) ^ flip);
    out.y = a.y + __int_as_float(__float_as_int(t.y) ^ flip);
    return out;
}

// ---------------------------------------------------------------------------
// complementary_fft_kernel — register-resident stages (spans 1..16)
//
//   Each warp handles 32 consecutive samples and executes butterfly stages
//   with spans 1, 2, 4, 8, 16 entirely in registers:
//     - input is loaded with a full bit-reversed permutation over the
//       transform index (logN bits), so the combined pipeline yields a
//       natural-order spectrum without any extra permutation pass
//     - the span-16 stage is the T_i / T_{i+16} complementary-pair stage
//
//   in        : input samples
//   out       : partially (chunk mode) or fully (logN == 5) transformed data
//   numChunks : number of 32-sample warps (= N / 32)
//   logN      : log2 of the full transform length; 5 in chunk mode
// ---------------------------------------------------------------------------
__global__ void complementary_fft_kernel(const float2* __restrict__ in,
                                         float2* __restrict__ out,
                                         int numChunks, int logN)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= numChunks) return;

    const int base = warp << 5;

    // Bit-reversed load over the transform-local index (logN bits); index
    // bits above logN (independent transforms in chunk mode) pass through
    // unchanged. For logN == 5 this reduces to base + brev5(lane).
    const unsigned g   = (unsigned)(base + lane);
    const unsigned rev = (g & ~((1u << logN) - 1u))
                       | (__brev(g) >> (32 - logN));
    float2 v = in[rev];

    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        const unsigned mask = 1u << s;                    // span: 1,2,4,8,16

        // Twiddle W = exp(-2*pi*i*k/(2*mask)), k = index within the group.
        // Groups are 2*mask-aligned, so k depends only on the low lane bits.
        const int   k     = lane & (int)(mask - 1u);
        const float angle = -2.0f * CUDART_PI_F * (float)k / (float)(2u * mask);
        float2 w;
        sincosf(angle, &w.y, &w.x);                       // (cos, sin)

        v = warp_butterfly(v, w, mask, lane);
    }

    out[base + lane] = v;
}

// ---------------------------------------------------------------------------
// butterfly_stage_kernel — one cross-warp butterfly stage (span >= 32)
//
//   Full-FFT mode only. Stages with span >= 32 pair samples that live in
//   different warps, so the exchange must go through global memory — the
//   same multi-pass structure any generic plan uses. One thread computes
//   one whole butterfly (both outputs), N/2 threads per pass.
//
//   data : the array being transformed in place
//   half : N / 2 (number of butterflies in this stage)
//   s    : stage index; span mask = 1 << s, group length = 2 << s
// ---------------------------------------------------------------------------
__global__ void butterfly_stage_kernel(float2* __restrict__ data,
                                       int half, int s)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half) return;

    const unsigned mask = 1u << s;
    const int k = tid & (int)(mask - 1u);                 // twiddle index
    const int i = ((tid >> s) << (s + 1)) | k;            // upper sample
    const int j = i + (int)mask;                          // lower sample

    const float angle = -2.0f * CUDART_PI_F * (float)k / (float)(2u * mask);
    float2 w;
    sincosf(angle, &w.y, &w.x);

    const float2 a = data[i];
    const float2 b = data[j];
    const float2 t = cmul(w, b);

    data[i] = make_float2(a.x + t.x, a.y + t.y);
    data[j] = make_float2(a.x - t.x, a.y - t.y);
}

// ---------------------------------------------------------------------------
// Host-side naive DFT of one 32-point chunk (chunk-mode reference).
// ---------------------------------------------------------------------------
static void reference_dft32(const float2* in, float2* out, int chunk)
{
    const float2* src = in + chunk * 32;
    for (int n = 0; n < 32; ++n) {
        double re = 0.0, im = 0.0;
        for (int t = 0; t < 32; ++t) {
            const double ang = -2.0 * M_PI * (double)(n * t) / 32.0;
            re += (double)src[t].x * cos(ang) - (double)src[t].y * sin(ang);
            im += (double)src[t].x * sin(ang) + (double)src[t].y * cos(ang);
        }
        out[n] = make_float2((float)re, (float)im);
    }
}

// ---------------------------------------------------------------------------
// Host-side iterative radix-2 FFT in double precision (full-FFT reference).
// ---------------------------------------------------------------------------
static void reference_fft(const float2* in, float2* out, int N, int logN)
{
    double* re = (double*)malloc((size_t)N * sizeof(double));
    double* im = (double*)malloc((size_t)N * sizeof(double));

    // Bit-reversed copy.
    for (int i = 0; i < N; ++i) {
        unsigned x = (unsigned)i, r = 0;
        for (int b = 0; b < logN; ++b) { r = (r << 1) | (x & 1u); x >>= 1; }
        re[r] = (double)in[i].x;
        im[r] = (double)in[i].y;
    }

    // Iterative DIT stages.
    for (int len = 2; len <= N; len <<= 1) {
        const int halfL = len >> 1;
        const double step = -2.0 * M_PI / (double)len;
        for (int i0 = 0; i0 < N; i0 += len) {
            for (int k = 0; k < halfL; ++k) {
                const double c = cos(step * (double)k);
                const double s = sin(step * (double)k);
                const int u = i0 + k, v = u + halfL;
                const double tr = re[v] * c - im[v] * s;
                const double ti = re[v] * s + im[v] * c;
                re[v] = re[u] - tr;  im[v] = im[u] - ti;
                re[u] += tr;         im[u] += ti;
            }
        }
    }

    for (int i = 0; i < N; ++i) out[i] = make_float2((float)re[i], (float)im[i]);
    free(re);
    free(im);
}

int main(int argc, char** argv)
{
    // Usage: complementary_fft [numChunks] [fullFft]
    //   numChunks : number of 32-sample warps (default 2048 -> N = 65536)
    //   fullFft   : 0 = independent 32-point FFTs (default)
    //               1 = one N-point FFT (numChunks must be a power of two)
    const int numChunks = (argc > 1) ? atoi(argv[1]) : 2048;
    const int fullFft   = (argc > 2) ? atoi(argv[2]) : 0;

    if (fullFft && (numChunks & (numChunks - 1)) != 0) {
        fprintf(stderr, "full-FFT mode requires numChunks to be a power of two\n");
        return EXIT_FAILURE;
    }

    int logChunks = 0;
    while ((1 << logChunks) < numChunks) ++logChunks;
    const int logN  = fullFft ? 5 + logChunks : 5;
    const int N     = numChunks * 32;
    const size_t bytes = (size_t)N * sizeof(float2);

    printf("[complementary_fft] %s, N = %d complex points (%.2f MB)\n",
           fullFft ? "full-FFT mode" : "chunk mode (independent 32-point FFTs)",
           N, (double)bytes / (1024.0 * 1024.0));

    // --- Input initialization: a deterministic two-tone signal ---
    // Chunk mode: tones at bins 3 and 7 of each 32-point chunk.
    // Full mode : tones at bins 3*numChunks and 7*numChunks of the N-point
    //             transform (same normalized frequencies).
    float2* h_in  = (float2*)malloc(bytes);
    float2* h_out = (float2*)malloc(bytes);
    for (int g = 0; g < N; ++g) {
        const int unit = fullFft ? N : 32;
        const int t    = fullFft ? g : (g & 31);
        const int f1   = fullFft ? 3 * numChunks : 3;
        const int f2   = fullFft ? 7 * numChunks : 7;
        const float ph1 = 2.0f * CUDART_PI_F * (float)f1 * (float)t / (float)unit;
        const float ph2 = 2.0f * CUDART_PI_F * (float)f2 * (float)t / (float)unit
                        + 0.01f * (float)(g >> 5);
        h_in[g] = make_float2(cosf(ph1) + 0.5f * cosf(ph2),
                              sinf(ph1) + 0.5f * sinf(ph2));
    }

    float2 *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int warpsPerBlock = 8;
    const int blocks = (numChunks + warpsPerBlock - 1) / warpsPerBlock;
    const dim3 grid(blocks), block(warpsPerBlock * 32);
    const int stageThreads = 256;
    const int stageBlocks  = (N / 2 + stageThreads - 1) / stageThreads;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Warm-up: register-resident stages, then the cross-warp global passes.
    complementary_fft_kernel<<<grid, block>>>(d_in, d_out, numChunks, logN);
    for (int s = 5; s < logN; ++s)
        butterfly_stage_kernel<<<stageBlocks, stageThreads>>>(d_out, N / 2, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run.
    CUDA_CHECK(cudaEventRecord(t0));
    complementary_fft_kernel<<<grid, block>>>(d_in, d_out, numChunks, logN);
    for (int s = 5; s < logN; ++s)
        butterfly_stage_kernel<<<stageBlocks, stageThreads>>>(d_out, N / 2, s);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // DRAM traffic: the warp kernel is 1 load + 1 store per element; each
    // cross-warp stage adds one more read+write pass. Passes = logN - 4
    // instead of the logN passes of a fully memory-staged plan.
    const int passes = fullFft ? (logN - 4) : 1;
    const double gbps = (2.0 * (double)bytes * (double)passes)
                      / ((double)ms * 1e-3) / 1e9;
    printf("[kernel] %.4f ms, %d DRAM pass(es) -> effective bandwidth %.2f GB/s\n",
           ms, passes, gbps);

    // --- Verification ---
    double maxErr = 0.0, maxMag = 0.0;
    if (!fullFft) {
        float2* h_ref = (float2*)malloc(32 * sizeof(float2));
        for (int c = 0; c < numChunks; c += (numChunks > 7 ? numChunks / 7 : 1)) {
            reference_dft32(h_in, h_ref, c);
            for (int n = 0; n < 32; ++n) {
                const double dr = (double)h_out[c * 32 + n].x - (double)h_ref[n].x;
                const double di = (double)h_out[c * 32 + n].y - (double)h_ref[n].y;
                const double e  = sqrt(dr * dr + di * di);
                if (e > maxErr) maxErr = e;
            }
        }
        free(h_ref);
        maxMag = 1.0;   // absolute tolerance for the small chunk transforms
    } else {
        float2* h_ref = (float2*)malloc(bytes);
        reference_fft(h_in, h_ref, N, logN);
        for (int i = 0; i < N; ++i) {
            const double dr = (double)h_out[i].x - (double)h_ref[i].x;
            const double di = (double)h_out[i].y - (double)h_ref[i].y;
            const double e  = sqrt(dr * dr + di * di);
            const double m  = sqrt((double)h_ref[i].x * h_ref[i].x
                                 + (double)h_ref[i].y * h_ref[i].y);
            if (e > maxErr) maxErr = e;
            if (m > maxMag) maxMag = m;
        }
        free(h_ref);
    }

    const double rel = maxErr / (maxMag > 0.0 ? maxMag : 1.0);
    const bool ok = fullFft ? (rel < 1e-3) : (maxErr < 1e-3);
    printf("[verify] max |error| = %.3e (relative %.3e) vs host reference -> %s\n",
           maxErr, rel, ok ? "PASS" : "FAIL");

    // --- Cleanup ---
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
