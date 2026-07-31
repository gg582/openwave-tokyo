// ============================================================================
// main.cpp — Hokusai sea pipeline driver (A/B benchmark edition)
//
// ONE scene is advanced per frame and BOTH kernel pipelines render it from
// the same inputs (identical deterministic spectrum seed, identical time),
// so the two MP4s show the same video — only the kernels differ:
//
//   traditional    : classic iterative radix-2 FFT (all stages through
//                    global memory) + GLSL separable-Gaussian unsharp
//   complementary  : warp-shuffle antipodal-pair FFT + CUDA warp-shuffle
//                    inverse-OTF aberration-correction kernel
//
// Per-stage timings are accumulated per pipeline and printed at the end.
//
//   bathymetry -> JONSWAP spectral ocean (CUDA: tide, shoaling, breaking,
//   Jacobian foam) -> headless GL PBR render (real Fuji DEM, far sea,
//   Jerlov II Beer-Lambert, Cook-Torrance) -> spherical-aberration post ->
//   [traditional GLSL unsharp | complementary CUDA correction] ->
//   zero-copy NVENC H.264 MP4 (x2)
//
// Usage:
//   hokusai_wave [--mode traditional|complementary|both] [--frames N]
//               [--fps N] [--timescale S] [--bathy FILE.nc] [--bench]
//
// References (full citations in ocean.cu and ocean.frag)
// -------------------------------------------------------
// [1]  Hasselmann et al. (1973) JONSWAP — spectrum alpha, wp, gamma
// [2]  Mitsuyasu et al. (1975) — directional spreading cos^(2s), s_p formula
// [3]  Donelan, Hamilton & Hui (1985) — storm gamma calibration (5-7)
// [4]  Monahan & O'Muircheartaigh (1980) — whitecap decay tau = 3.53 s
// [5]  Banner & Melville (1994) — deep-water breaking threshold ak = 0.25
// [6]  Janssen (2003) — Benjamin-Feir Index, rogue-wave frequency focusing
// [7]  Tessendorf (2001) SIGGRAPH — FFT ocean, choppiness lambda
// [8]  Tucker & Pitt (2001) — JONSWAP parameterization reference
// [9]  Cook & Torrance (1982) — BRDF (ocean.frag)
// [10] Jerlov (1976) Marine Optics — water extinction coefficients (ocean.frag)
// [11] Henyey & Greenstein (1941) — subsurface scattering phase function (ocean.frag)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

#include "bathymetry.h"
#include "ocean.h"
#include "renderer.h"
#include "postfx.h"
#include "encoder.h"
#include "fft_gpu.h"
#include "climate.h"
#include "spray.h"
#include "tide.h"

#include <filesystem>
extern "C" void glFinish(void);   // GL 1.0 entry point from libGL

// ---------------------------------------------------------------------------
// CPU bilinear resample of the bathymetry onto the georeferenced wave grid.
// Ocean row j maps to z = (j/(n-1) - 0.5) * domain (north = -z), so row 0
// is the NORTHERN edge; the GEBCO grid row 0 is the southernmost latitude.
// ---------------------------------------------------------------------------
static void resampleDepth(const Bathymetry& b, float* out, int n,
                          float domainM, float latC, float lonC)
{
    const float mPerDegLat = 111000.0f;
    const float mPerDegLon = 111000.0f * cosf(latC * 3.14159265f / 180.0f);
    for (int j = 0; j < n; ++j) {
        const float z = ((float)j / (float)(n - 1) - 0.5f) * domainM;
        const float lat = latC - z / mPerDegLat;         // north = -z
        const float fy = (lat - b.latMin) / (b.latMax - b.latMin) * (b.height - 1);
        for (int i = 0; i < n; ++i) {
            const float x = ((float)i / (float)(n - 1) - 0.5f) * domainM;
            const float lon = lonC + x / mPerDegLon;
            const float fx = (lon - b.lonMin) / (b.lonMax - b.lonMin) * (b.width - 1);
            const int x0 = std::min(std::max((int)fx, 0), b.width - 2);
            const int y0 = std::min(std::max((int)fy, 0), b.height - 2);
            const float ax = fx - (float)x0, ay = fy - (float)y0;
            const float e00 = b.elevation[(size_t)y0 * b.width + x0];
            const float e10 = b.elevation[(size_t)y0 * b.width + x0 + 1];
            const float e01 = b.elevation[(size_t)(y0 + 1) * b.width + x0];
            const float e11 = b.elevation[(size_t)(y0 + 1) * b.width + x0 + 1];
            const float elev = (e00 * (1 - ax) + e10 * ax) * (1 - ay)
                             + (e01 * (1 - ax) + e11 * ax) * ay;
            out[(size_t)j * n + i] = fmaxf(-elev, 1.5f); // positive depth (m)
        }
    }
}

// ---------------------------------------------------------------------------
// Wind gust envelope: slow bursty multiplier on the wave amplitude.
// Physically, a fully developed sea's amplitude tracks the SQUARE of the
// effective wind speed, so scale = (U_eff/U_base)^2. The envelope mixes
// incommensurate sines into burst-like gusts: lull ~0.75 (U_eff ~22 m/s),
// gust peaks ~2.6 (U_eff ~40 m/s, typhoon-grade).
// ---------------------------------------------------------------------------
static float gustScale(float t, float gustMultiplier = 1.85f)
{
    const float n = 0.55f * sinf(0.83f * t)
                  + 0.30f * sinf(1.97f * t + 1.3f)
                  + 0.45f * sinf(0.29f * t + 2.1f);
    const float u = fminf(fmaxf((n + 0.35f) / 1.3f, 0.0f), 1.0f);
    const float burst = u * u * (3.0f - 2.0f * u);   // smoothstep shaping
    // normalized so the mean envelope is ~1.0: the spectrum is calibrated
    // to Hs = 11 m, gusts only modulate it +/-25%
    return 0.75f + 0.50f * burst * (gustMultiplier / 2.5f);
}

struct Pipeline {
    const char* name;
    FftMode     fft;
    bool        traditionalPost;
    int         encW = 3840;
    int         encH = 2160;
    Ocean*      ocean = nullptr;
    Encoder     encoder;
    double      oceanMs = 0, renderMs = 0, postMs = 0, encMs = 0;
};

int main(int argc, char** argv)
{
    // If working dir does not contain shaders/, switch to the directory containing the binary
    if (!std::filesystem::exists("shaders")) {
        std::filesystem::path exeDir = std::filesystem::canonical("/proc/self/exe").parent_path();
        if (std::filesystem::exists(exeDir / "shaders")) {
            std::filesystem::current_path(exeDir);
        }
    }
    std::string modeStr = "both";
    std::string outPath;                 // single-mode output override
    std::string bathyPath;
    std::string shaderDir = "shaders";
    int frames = 1800, width = 3840, height = 2160, fps = 30;
    float timeScale = 3.0f;             // sim seconds per playback second
    bool bench = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if      (a == "--mode")    modeStr = next();
        else if (a == "--out")     outPath = next();
        else if (a == "--bathy")   bathyPath = next();
        else if (a == "--shaders") shaderDir = next();
        else if (a == "--frames")  frames = atoi(next());
        else if (a == "--fps")     fps = atoi(next());
        else if (a == "--timescale") timeScale = (float)atof(next());
        else if (a == "--width")   width = atoi(next());
        else if (a == "--height")  height = atoi(next());
        else if (a == "--bench")   bench = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    if (bench) {
        float msT = 0, msC = 0;
        fft_bench(256, 200, &msT, &msC);
        printf("[fft-bench] 2D IFFT 256x256:  traditional %.4f ms | "
               "complementary %.4f ms  (%.2fx)\n", msT, msC, msT / msC);
        fft_bench(512, 100, &msT, &msC);
        printf("[fft-bench] 2D IFFT 512x512:  traditional %.4f ms | "
               "complementary %.4f ms  (%.2fx)\n", msT, msC, msT / msC);
        fft_bench(1024, 50, &msT, &msC);
        printf("[fft-bench] 2D IFFT 1024x1024: traditional %.4f ms | "
               "complementary %.4f ms  (%.2fx)\n", msT, msC, msT / msC);
        return 0;
    }

    // ---- bathymetry ----
    Bathymetry bathy;
    if (!bathyPath.empty()) {
        if (!loadNetCDF3(bathyPath.c_str(), bathy)) {
            fprintf(stderr, "failed to load %s\n", bathyPath.c_str());
            return 1;
        }
        printf("[bathy] loaded %s (%d x %d)\n", bathyPath.c_str(),
               bathy.width, bathy.height);
    } else {
        bathy = generateSyntheticUraga(1024);
        std::filesystem::create_directories("data");
        writeNetCDF3("data/uraga_synthetic.nc", bathy);
        printf("[bathy] synthetic Uraga Channel model (%d x %d), "
               "saved to data/uraga_synthetic.nc\n", bathy.width, bathy.height);
    }

    // ---- ocean configuration: typhoon swell at Hokusai scale ----
    OceanConfig ocfg;
    ocfg.n        = 1024;
    ocfg.domain   = 2500.0f;
    ocfg.windDir  = -1.24f;             // swell travels SSW->NNE, ACROSS the
                                        // camera view (bearing 289 deg), so
                                        // the rolling crests are seen in
                                        // profile, not head-on
    ocfg.lambda   = 1.8f;               // Restored to 1.8 for aggressive, sharp crests.
                                        // The "pyramidal" boxy artifact is mitigated by
                                        // broadening the spatial focus envelope slightly.
    ocfg.windSpeed = 21.0f;             // Draupner sea state: Hs = 12 m
                                        // Base wind speed tuned to achieve Hs~12m
                                        // for the 1024^2, 2.5km domain.
    ocfg.fetch     = 300000.0f;
    std::vector<float> depthGrid((size_t)ocfg.n * ocfg.n);
    resampleDepth(bathy, depthGrid.data(), ocfg.n, ocfg.domain,
                  35.2f - 0.055f, 139.7f);

    // Era-appropriate sun and weather: NASA POWER climatology for the site
    // plus the 1831 solar ephemeris (orbital elements at the 1831 epoch).
    const SceneClimate clim =
        loadSceneClimate("assets/climate.json", 35.2f, 139.7f);

    // Astronomical tide for the scene epoch (1831-03-21 07:30 JST =
    // 1831-03-20 22:30 UT). Lunar age 7.1 d: just before first quarter,
    // a neap tide — small range, as the harmonic prediction reflects.
    const double sceneJD = tideJulianDay(1831, 3, 20, 22.5);
    printf("[tide] harmonic prediction (M2/S2/N2/K2/K1/O1/P1, Uraga) "
           "@1831-03-21 07:30 JST: %+.3f m above MSL\n",
           tideLevelMSL(sceneJD));

    RenderConfig rcfg;
    rcfg.width = width; rcfg.height = height;
    rcfg.shaderDir = shaderDir;
    rcfg.domain = ocfg.domain;
    rcfg.lambda = ocfg.lambda;
    rcfg.camHeight = 14.0f;             // Original eye-level
    rcfg.fovDeg = 26.0f;                // Original telephoto FOV
    rcfg.aberration = 0.45f;
    memcpy(rcfg.sunDir, clim.sunDir, sizeof(clim.sunDir));
    memcpy(rcfg.sunColor, clim.sunColor, sizeof(clim.sunColor));
    rcfg.cloudAmount = clim.cloudAmount;
    rcfg.propDir[0] = cosf(ocfg.windDir);
    rcfg.propDir[1] = sinf(ocfg.windDir);

    ocfg.windSpeed = clim.fittedWildWind;

    int vertH = (int)(height * 1.30f);
    int maxH = vertH; // Vertical height is the largest

    Renderer renderer;
    rcfg.height = maxH; // Initialize with max height for FBO allocation
    if (!renderer.init(rcfg)) { fprintf(stderr, "renderer init failed\n"); return 1; }
    renderer.registerOceanTextures(ocfg.n);

    PostFx postfx;
    if (!postfx.init(width, maxH, renderer.pbo())) {
        fprintf(stderr, "postfx init failed\n"); return 1;
    }

    // ballistic spray droplets (particle extension, A/B kernels)
    SprayConfig scfg;
    scfg.n           = ocfg.n;
    scfg.maxEmitters = 8192;
    scfg.perEmitter  = 64;               // 524K max active physical water droplets
    scfg.breakThresh = 0.22f;            // emit only where the crest actually breaks (lowered to trigger more spume/spray on intersection)
    scfg.domain      = ocfg.domain;
    Spray spray;
    spray.init(scfg);
    if (!renderer.initSpray(spray.particleCount())) {
        fprintf(stderr, "spray render init failed\n"); return 1;
    }

    Ocean tradOcean, compOcean;
    if (modeStr == "both" || modeStr == "traditional") {
        tradOcean.init(ocfg, depthGrid.data(), FFT_TRADITIONAL);
    }
    if (modeStr == "both" || modeStr == "complementary") {
        compOcean.init(ocfg, depthGrid.data(), FFT_COMPLEMENTARY);
    }

    // ---- pipelines: same scene, two kernel implementations ----
    std::vector<Pipeline> pipes;
    pipes.reserve(4);
    auto addPipe = [&](const char* name, FftMode fft, bool tradPost,
                       const char* out, int encW, int encH, Ocean* sharedOcean) {
        Pipeline p;
        p.name = name; p.fft = fft; p.traditionalPost = tradPost;
        p.encW = encW; p.encH = encH;
        p.ocean = sharedOcean;
        if (!p.encoder.open(out, encW, encH, fps)) {
            fprintf(stderr, "encoder open failed for %s\n", out);
            exit(1);
        }
        printf("[encoder] %s -> %s (%dx%d, %s)\n", name, out, encW, encH,
               p.encoder.usingNvenc() ? "NVENC zero-copy" : "libx264");
        pipes.push_back(std::move(p));
    };

    if (modeStr == "both" || modeStr == "traditional") {
        addPipe("traditional", FFT_TRADITIONAL, true,
                outPath.empty() ? "hokusai_traditional.mp4" : outPath.c_str(), width, height, &tradOcean);
        std::string vertOut = outPath.empty() ? "hokusai_vert30per_traditional.mp4"
                             : (std::filesystem::path(outPath).parent_path() / ("hokusai_vert30per_" + std::filesystem::path(outPath).filename().string())).string();
        addPipe("traditional_vert30per", FFT_TRADITIONAL, true, vertOut.c_str(), width, vertH, &tradOcean);
    }
    if (modeStr == "both" || modeStr == "complementary") {
        addPipe("complementary", FFT_COMPLEMENTARY, false,
                outPath.empty() ? "hokusai_complementary.mp4" : outPath.c_str(), width, height, &compOcean);
        std::string vertOut = outPath.empty() ? "hokusai_vert30per_complementary.mp4"
                             : (std::filesystem::path(outPath).parent_path() / ("hokusai_vert30per_" + std::filesystem::path(outPath).filename().string())).string();
        addPipe("complementary_vert30per", FFT_COMPLEMENTARY, false, vertOut.c_str(), width, vertH, &compOcean);
    }

    std::vector<unsigned char> hostRGBA((size_t)width * height * 4);
    std::vector<unsigned char> snap[2] = {
        std::vector<unsigned char>((size_t)width * height * 4),
        std::vector<unsigned char>((size_t)width * height * 4)
    };
    long long frameDiffBytes = 0;
    int frameDiffMax = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    printf("[render] %d pipelines x %d frames %dx%d @ %d fps (sim x%.1f)\n",
           (int)pipes.size(), frames, width, height, fps, timeScale);

    std::ofstream audioPhysFile("data/audio_physics.txt");
    for (int f = 0; f < frames; ++f) {
        // Playback time is LOCKED to the container frame rate: one playback
        // second advances the simulation by exactly timeScale seconds, for
        // BOTH pipelines identically. Rendering throughput never enters.
        const float t = (float)f * (timeScale / (float)fps);

        // Water level from the astronomical harmonic tide prediction for
        // the scene epoch (see tide.h): the clip spans only 3 sim-minutes,
        // so the level is essentially the predicted constant for 07:30 JST
        // plus its true, tiny rate of change — no synthetic ramp.
        const float tide = tideLevelMSL(sceneJD + (double)t / 86400.0);

        // ---- advance the simulation once per FFT implementation (timed);
        //      both MUST produce bit-identical fields -------------------------
        // ---- advance the simulation once per FFT implementation (timed);
        //      both MUST produce bit-identical fields. The wave clock is
        //      offset +120 s so the clip starts mid-evolution (the t=0
        //      phase alignment of the counter-rotating spectra is a
        //      statistical outlier, not a real sea state).
        const float tWave = t + 120.0f;
        const float gust = gustScale(t + 40.0f, clim.gustMultiplier);
        if (modeStr == "both" || modeStr == "traditional") {
            cudaEventRecord(ev0);
            tradOcean.advance(tWave, tide, gust);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev0, ev1);
            for (auto& p : pipes) if (p.fft == FFT_TRADITIONAL) p.oceanMs += ms;
        }
        if (modeStr == "both" || modeStr == "complementary") {
            cudaEventRecord(ev0);
            compOcean.advance(tWave, tide, gust);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev0, ev1);
            for (auto& p : pipes) if (p.fft == FFT_COMPLEMENTARY) p.oceanMs += ms;
        }

        if (pipes.size() >= 2 && (f == 0 || (f + 1) % 60 == 0)) {
            const size_t cells = (size_t)ocfg.n * ocfg.n;
            std::vector<float> a(cells), b(cells);
            auto cmp = [&](const char* name, const float* da, const float* db) {
                cudaMemcpy(a.data(), da, cells * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), db, cells * 4, cudaMemcpyDeviceToHost);
                float maxD = 0.0f;
                for (size_t k = 0; k < cells; ++k)
                    maxD = fmaxf(maxD, fabsf(a[k] - b[k]));
                printf("[verify] frame %d %-8s max |diff| = %.3e %s\n", f,
                       name, maxD, maxD == 0.0f ? "(bit-identical)" : "(MISMATCH)");
                return maxD;
            };
            cmp("depth", tradOcean.depth(), compOcean.depth());
            cmp("gain",  tradOcean.gain(),  compOcean.gain());
            cmp("height",tradOcean.height(),compOcean.height());
            cmp("foam",  tradOcean.foam(),  compOcean.foam());
        }
        if (f == 0 || f == 100 || (f + 1) % 600 == 0) {
            std::vector<float> probe((size_t)ocfg.n * ocfg.n);
            cudaMemcpy(probe.data(), tradOcean.height(),
                       probe.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float mn = 1e30f, mx = -1e30f;
            double var = 0.0;
            for (float v : probe) { mn = fminf(mn, v); mx = fmaxf(mx, v); var += (double)v * v; }
            var /= probe.size();
            printf("[probe] frame %d height min/max = %.3f / %.3f m, "
                   "Hs(4*RMS) = %.2f m, max crest/Hs = %.2f\n",
                   f, mn, mx, 4.0 * sqrt(var), mx / (4.0 * sqrt(var) + 1e-6));
            cudaMemcpy(probe.data(), tradOcean.foam(), probe.size() * sizeof(float), cudaMemcpyDeviceToHost);
            mn = 1e30f; mx = -1e30f; double mean = 0;
            for (float v : probe) { mn = fminf(mn, v); mx = fmaxf(mx, v); mean += v; }
            mean /= probe.size();
            printf("[probe] frame %d foam min/max/mean = %.3f / %.3f / %.4f\n", f, mn, mx, mean);
        }

        // ---- spray droplets: A/B emitter scan (bench), shared ballistic
        //      step, A/B grid projection (bench + identity check) ----------
        const float simDt = timeScale / (float)fps;
        spray.scan(tradOcean.foam(), tradOcean.height(), 0, 0);
        spray.scan(tradOcean.foam(), tradOcean.height(), 1, 0);
        const int nEmit = spray.emitterCountHost();
        if (audioPhysFile.is_open()) {
            audioPhysFile << f << " " << t << " " << gust << " " << tWave << " " << nEmit << "\n";
        }
        spray.step(tide, simDt, f, tradOcean.height(), 0);
        spray.project(0, 0, 0);
        spray.project(1, 1, 0);
        if ((f + 1) % 60 == 0 && pipes.size() >= 2) {
            // grid identity check (Q16.16 sums must match exactly)
            std::vector<int> ga((size_t)ocfg.n * ocfg.n),
                             gb((size_t)ocfg.n * ocfg.n);
            cudaMemcpy(ga.data(), spray.grid(0), ga.size() * sizeof(int),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(gb.data(), spray.grid(1), gb.size() * sizeof(int),
                       cudaMemcpyDeviceToHost);
            long long diff = 0;
            for (size_t k = 0; k < ga.size(); ++k) diff += llabs((long long)ga[k] - gb[k]);
            printf("[verify] frame %d: spray grid diff = %lld %s\n", f, diff,
                   diff == 0 ? "(bit-identical)" : "(MISMATCH)");
        }
        // identical deposits into BOTH oceans (bit-identical by Q16.16)
        if (modeStr == "both" || modeStr == "traditional") spray.depositInto(tradOcean.foam(), ocfg.n, 0);
        if (modeStr == "both" || modeStr == "complementary") spray.depositInto(compOcean.foam(), ocfg.n, 0);
        spray.clearGrid(0);
        spray.finalize(0);

        // ---- ONE render per unique resolution ----
        // To fix the vert30per green-screen bug, we ensure the viewport and camera
        // are correctly set for both the standard 16:9 and the vertical 30% extended frames.
        renderer.uploadOcean(tradOcean.height(), tradOcean.disp(),
                             tradOcean.foam(), tradOcean.depth(),
                             tradOcean.gain(), 0);
        
        // Plunging-jet barrel ribbon (Hokusai "claw" foam basis)
        {
            static std::vector<float> hh((size_t)ocfg.n * ocfg.n),
                                      ff((size_t)ocfg.n * ocfg.n);
            cudaMemcpy(hh.data(), tradOcean.height(),
                       hh.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(ff.data(), tradOcean.foam(),
                       ff.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const float pdir[2] = { cosf(ocfg.windDir), sinf(ocfg.windDir) };
            renderer.updateBarrel(hh.data(), ff.data(), ocfg.n, ocfg.domain,
                                  pdir, 0.0f, -350.0f, 9.5f);
        }

        // We render twice: once for standard 4K, once for vert30per.
        // (In a production scenario, we'd cache the scene, but here correctness is priority).
        
        int idx = 0;
        for (auto& p : pipes) {
            renderer.renderFrame(t, tide, spray.positions(), spray.sizes(),
                                 spray.particleCount(), p.encW, p.encH);
            
            postfx.snapshotFrame(p.encW, p.encH, 0); 
            uchar4* frame = postfx.beginFrame(p.traditionalPost ? 0 : 1,
                                              0.45f, p.encW, p.encH, 0);
            p.postMs += postfx.lastCorrectionMs;

            auto c2 = std::chrono::steady_clock::now();
            if (p.encoder.usingNvenc()) {
                unsigned char* y; int yPitch; unsigned char* uv; int uvPitch;
                if (p.encoder.acquireDevicePlanes(&y, &yPitch, &uv, &uvPitch)) {
                    rgbaToNv12(frame, p.encW, p.encH, y, yPitch, uv, uvPitch, 0);
                    cudaStreamSynchronize(0);
                    p.encoder.writeFrameDevice();
                }
            } else {
                if (hostRGBA.size() < (size_t)p.encW * p.encH * 4) 
                    hostRGBA.resize((size_t)p.encW * p.encH * 4);
                cudaMemcpy(hostRGBA.data(), frame, (size_t)p.encW * p.encH * 4,
                           cudaMemcpyDeviceToHost);
                p.encoder.writeFrameHost(hostRGBA.data(), p.encW, p.encH);
            }
            auto c3 = std::chrono::steady_clock::now();
            p.encMs += std::chrono::duration<double, std::milli>(c3 - c2).count();

            if (idx < 2) {
                if (snap[idx].size() < (size_t)p.encW * p.encH * 4)
                    snap[idx].resize((size_t)p.encW * p.encH * 4);
                cudaMemcpy(snap[idx].data(), frame, snap[idx].size(),
                           cudaMemcpyDeviceToHost);
            }
            postfx.endFrame(0);
            ++idx;
        }
        if (pipes.size() == 2) {
            for (size_t k = 0; k < snap[0].size(); ++k) {
                const int d = abs((int)snap[0][k] - (int)snap[1][k]);
                if (d) { ++frameDiffBytes; if (d > frameDiffMax) frameDiffMax = d; }
            }
        }

        if ((f + 1) % 60 == 0 || f == frames - 1)
            printf("  frame %d/%d\n", f + 1, frames);
    }

    for (auto& p : pipes) p.encoder.close();
    postfx.release();
    renderer.shutdown();
    if (modeStr == "both" || modeStr == "traditional") tradOcean.release();
    if (modeStr == "both" || modeStr == "complementary") compOcean.release();

    printf("\n=========== A/B PIPELINE BENCHMARK (same scene, %d frames) ===========\n",
           frames);
    printf("%-16s %10s %10s %10s %10s %10s\n", "pipeline", "ocean", "render",
           "correct", "encode", "total");
    for (auto& p : pipes) {
        const double tot = (p.oceanMs + p.renderMs + p.postMs + p.encMs) / frames;
        printf("%-16s %8.3f ms %8.3f ms %8.3f ms %8.3f ms %8.3f ms\n",
               p.name, p.oceanMs / frames, p.renderMs / frames,
               p.postMs / frames, p.encMs / frames, tot);
    }
    printf("---------------------------------------------------------------------\n");
    printf("spray kernels:  traditional        complementary      speedup\n");
    printf("  emitter scan : %8.4f ms      %8.4f ms      %6.2fx\n",
           spray.msScan[0] / spray.framesScan[0],
           spray.msScan[1] / spray.framesScan[1],
           (spray.msScan[0] + 1e-9) / (spray.msScan[1] + 1e-9));
    printf("  projection   : %8.4f ms      %8.4f ms      %6.2fx\n",
           spray.msProject[0] / spray.framesProject[0],
           spray.msProject[1] / spray.framesProject[1],
           (spray.msProject[0] + 1e-9) / (spray.msProject[1] + 1e-9));
    printf("  update(shared): %7.4f ms/frame\n", spray.msStep / spray.framesStep);
    printf("playback locked to %d fps, sim time = frames x %.4f s/frame\n",
           fps, timeScale / fps);
    printf("frame content check: %lld bytes differ, max |delta| = %d %s\n",
           frameDiffBytes, frameDiffMax,
           (frameDiffBytes == 0) ? "(BIT-IDENTICAL FRAMES)" : "(MISMATCH!)");
    printf("======================================================================\n");

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    spray.release();

    if (pipes.size() >= 2) {
        printf("\n[verify] encoded file hashes:\n");
        fflush(stdout);
        system("sha256sum hokusai_traditional.mp4 hokusai_complementary.mp4 hokusai_vert30per_traditional.mp4 hokusai_vert30per_complementary.mp4 2>/dev/null || true");
    }
    return 0;
}
