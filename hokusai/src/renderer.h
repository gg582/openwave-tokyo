// ============================================================================
// renderer.h — headless EGL + OpenGL 3.3 core renderer
//
// Draws the Hokusai composition (long-focus / low-height frustum) into an
// off-screen FBO chain:
//   scene FBO -> spherical-aberration FBO -> [traditional: 2-pass GLSL
//   unsharp] -> PBO (async readback) -> CUDA postfx / encoder
// Ocean field textures are updated straight from CUDA device memory through
// cudaGraphics GL interop (no host round trip).
// ============================================================================
#pragma once
#include <string>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct RenderConfig {
    int         width      = 1280;
    int         height     = 720;
    std::string shaderDir  = "shaders";
    std::string assetsDir  = "assets";
    float       domain     = 4000.0f;  // wave patch size (m)
    int         gridN      = 256;      // ocean texture resolution
    float       fovDeg     = 22.0f;    // long-focus lens (telephoto)
    float       camHeight  = 12.0f;    // low optical height (m)
    float       aberration = 0.7f;     // spherical aberration strength
    float       lambda     = 1.0f;     // horizontal choppiness (passed to uLambda)
    // era weather (NASA POWER + 1831 solar position), filled by main
    float       sunDir[3]   = {0.9f, 0.2f, 0.38f};
    float       sunColor[3] = {1.0f, 0.90f, 0.72f};
    float       cloudAmount = 0.61f;
    float       propDir[2]  = {0.325f, -0.946f};  // swell propagation (uv)
};

class Renderer {
public:
    bool init(const RenderConfig& cfg);
    void registerOceanTextures(int n);
    void uploadOcean(const float* h, const float2* disp, const float* foam,
                     const float* depth, const float* gain,
                     cudaStream_t stream);
    // Rebuild the plunging-jet barrel ribbon from the wave field
    // (host copies of the height and foam grids).
    void updateBarrel(const float* h, const float* foam, int n, float domain,
                      const float pdir[2], float bankX, float bankZ,
                      float phaseSpeed);
    // Renders the scene once: sky + terrain + far sea + ocean mesh, then
    // the spherical-aberration post pass, read back into the PBO.
    void renderFrame(float time, float tide, const float4* sprayPos,
                     const float* spraySizes, int sprayCount, int w, int h);
    // spray droplets: register the shared VBOs once, draw each frame from
    // CUDA device buffers (positions+alpha float4, diameters float)
    bool initSpray(int maxParticles);
    void drawSpray(const float4* devPositions, const float* devSizes,
                   int count, float time);
    void shutdown();

    unsigned pbo()    const { return pbo_; }
    int     width()   const { return cfg_.width; }
    int     height()  const { return cfg_.height; }

private:
    RenderConfig cfg_;

    // EGL
    void* dpy_ = nullptr;   // EGLDisplay
    void* ctx_ = nullptr;   // EGLContext
    void* surf_ = nullptr;  // EGLSurface

    // GL objects
    unsigned progBg_ = 0, progOcean_ = 0, progAber_ = 0, progUnsharp_ = 0;
    unsigned vaoGrid_ = 0, vboGrid_ = 0, iboGrid_ = 0;
    unsigned vaoQuad_ = 0;
    unsigned texHeight_ = 0, texDisp_ = 0, texFoam_ = 0, texDepth_ = 0, texGain_ = 0;
    unsigned fboScene_ = 0, texScene_ = 0;
    unsigned fboAber_  = 0, texAber_  = 0;
    unsigned fboTmp_   = 0, texTmp_   = 0;
    unsigned fboFinal_ = 0, texFinal_ = 0;
    unsigned pbo_ = 0;
    // CC0 material maps (ambientCG): foam albedo/normal/opacity, Fuji rock
    unsigned texFoamAlbedo_ = 0, texFoamNormal_ = 0, texFoamOpacity_ = 0;
    unsigned texFoamRough_ = 0;
    unsigned texRock_ = 0;
    // real Mount Fuji terrain (Terrarium DEM mesh)
    unsigned progTerrain_ = 0;
    unsigned vaoTerrain_ = 0, vboTerrain_ = 0, iboTerrain_ = 0;
    int      terrainIdxCount_ = 0;
    // barrel lip ribbon (plunging-jet surface, rebuilt per frame)
    unsigned progBarrel_ = 0;
    unsigned vaoBarrel_ = 0, vboBarrel_ = 0, iboBarrel_ = 0;
    int      barrelIdxCount_ = 0;
    // spray droplet point sprites
    unsigned progSpray_ = 0;
    unsigned vaoSpray_ = 0, vboSpray_ = 0, vboSpraySize_ = 0;
    cudaGraphicsResource* resSpray_ = nullptr;
    cudaGraphicsResource* resSpraySize_ = nullptr;
    int      sprayMaxP_ = 0;
    // far sea plane (fills the bay between the wave patch and the shore)
    unsigned progSeaFar_ = 0;
    unsigned vaoSeaFar_ = 0, vboSeaFar_ = 0, iboSeaFar_ = 0;
    int      seaFarIdxCount_ = 0;
    int      indexCount_ = 0;
    int      gridN_ = 0;
    int      fboW_ = 0, fboH_ = 0;

    // CUDA-GL interop handles
    cudaGraphicsResource* resHeight_ = nullptr;
    cudaGraphicsResource* resDisp_   = nullptr;
    cudaGraphicsResource* resFoam_   = nullptr;
    bool interopOk_ = false;

    float viewProj_[16];
    float camPos_[3]   = {0, 0, 0};
    float camDir_[3]   = {0, 0, -1};
    float camRight_[3] = {1, 0, 0};
    float camUp_[3]    = {0, 1, 0};

    unsigned compileProgram(const char* vertSrc, const char* fragSrc);
    unsigned makeColorTarget(int w, int h, unsigned* tex);
    void buildCamera(float time, int targetW, int targetH);
    bool     loadTerrain();
    bool     initSeaFar();
};
