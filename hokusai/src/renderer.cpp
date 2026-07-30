// ============================================================================
// renderer.cpp — headless EGL/OpenGL implementation
// ============================================================================
#include "renderer.h"
#include "texture_load.h"
#include "terrain.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

// ---------------------------------------------------------------------------
// Minimal GL 3.3 core loader (no GLEW dependency)
// ---------------------------------------------------------------------------
// two typedefs this glcorearb.h revision does not provide
#ifndef PFNGLINKPROGRAMPROC
typedef void (APIENTRYP PFNGLINKPROGRAMPROC)(GLuint program);
#endif
#ifndef PFNGLDRAWELEMENTSPROC
typedef void (APIENTRYP PFNGLDRAWELEMENTSPROC)(GLenum mode, GLsizei count,
                                               GLenum type, const void* indices);
#endif
#define GLFUNC_LIST(X) \
    X(PFNGLGENVERTEXARRAYSPROC,        glGenVertexArrays)        \
    X(PFNGLBINDVERTEXARRAYPROC,        glBindVertexArray)        \
    X(PFNGLGENBUFFERSPROC,             glGenBuffers)             \
    X(PFNGLBINDBUFFERPROC,             glBindBuffer)             \
    X(PFNGLBUFFERDATAPROC,             glBufferData)             \
    X(PFNGLBUFFERSUBDATAPROC,          glBufferSubData)          \
    X(PFNGLENABLEVERTEXATTRIBARRAYPROC,glEnableVertexAttribArray)\
    X(PFNGLVERTEXATTRIBPOINTERPROC,    glVertexAttribPointer)    \
    X(PFNGLCREATESHADERPROC,           glCreateShader)           \
    X(PFNGLSHADERSOURCEPROC,           glShaderSource)           \
    X(PFNGLCOMPILESHADERPROC,          glCompileShader)          \
    X(PFNGLGETSHADERIVPROC,            glGetShaderiv)            \
    X(PFNGLGETSHADERINFOLOGPROC,       glGetShaderInfoLog)       \
    X(PFNGLCREATEPROGRAMPROC,          glCreateProgram)          \
    X(PFNGLATTACHSHADERPROC,           glAttachShader)           \
    X(PFNGLINKPROGRAMPROC,             glLinkProgram)            \
    X(PFNGLGETPROGRAMIVPROC,           glGetProgramiv)           \
    X(PFNGLGETPROGRAMINFOLOGPROC,      glGetProgramInfoLog)      \
    X(PFNGLUSEPROGRAMPROC,             glUseProgram)             \
    X(PFNGLGETUNIFORMLOCATIONPROC,     glGetUniformLocation)     \
    X(PFNGLUNIFORM1IPROC,              glUniform1i)              \
    X(PFNGLUNIFORM1FPROC,              glUniform1f)              \
    X(PFNGLUNIFORM2FPROC,              glUniform2f)              \
    X(PFNGLUNIFORM3FPROC,              glUniform3f)              \
    X(PFNGLUNIFORMMATRIX4FVPROC,       glUniformMatrix4fv)       \
    X(PFNGLGENFRAMEBUFFERSPROC,        glGenFramebuffers)        \
    X(PFNGLBINDFRAMEBUFFERPROC,        glBindFramebuffer)        \
    X(PFNGLFRAMEBUFFERTEXTURE2DPROC,   glFramebufferTexture2D)   \
    X(PFNGLACTIVETEXTUREPROC,          glActiveTexture)          \
    X(PFNGLDRAWELEMENTSPROC,           glDrawElements)           \
    X(PFNGLGENRENDERBUFFERSPROC,       glGenRenderbuffers)       \
    X(PFNGLBINDRENDERBUFFERPROC,       glBindRenderbuffer)       \
    X(PFNGLRENDERBUFFERSTORAGEPROC,    glRenderbufferStorage)    \
    X(PFNGLFRAMEBUFFERRENDERBUFFERPROC,glFramebufferRenderbuffer)\
    X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus)\
    X(PFNGLGENERATEMIPMAPPROC,         glGenerateMipmap)

#define X(t, n) static t p_##n = nullptr;
GLFUNC_LIST(X)
#undef X

static bool loadGL()
{
    bool ok = true;
#define X(t, n) p_##n = (t)eglGetProcAddress(#n); ok = ok && (p_##n != nullptr);
    GLFUNC_LIST(X)
#undef X
    return ok;
}

// thin wrappers keep call sites readable
#define glGenVertexArrays        p_glGenVertexArrays
#define glBindVertexArray        p_glBindVertexArray
#define glGenBuffers             p_glGenBuffers
#define glBindBuffer             p_glBindBuffer
#define glBufferData             p_glBufferData
#define glBufferSubData          p_glBufferSubData
#define glEnableVertexAttribArray p_glEnableVertexAttribArray
#define glVertexAttribPointer    p_glVertexAttribPointer
#define glCreateShader           p_glCreateShader
#define glShaderSource           p_glShaderSource
#define glCompileShader          p_glCompileShader
#define glGetShaderiv            p_glGetShaderiv
#define glGetShaderInfoLog       p_glGetShaderInfoLog
#define glCreateProgram          p_glCreateProgram
#define glAttachShader           p_glAttachShader
#define glLinkProgram            p_glLinkProgram
#define glGetProgramiv           p_glGetProgramiv
#define glGetProgramInfoLog      p_glGetProgramInfoLog
#define glUseProgram             p_glUseProgram
#define glGetUniformLocation     p_glGetUniformLocation
#define glUniform1i              p_glUniform1i
#define glUniform1f              p_glUniform1f
#define glUniform2f              p_glUniform2f
#define glUniform3f              p_glUniform3f
#define glUniformMatrix4fv       p_glUniformMatrix4fv
#define glGenFramebuffers        p_glGenFramebuffers
#define glBindFramebuffer        p_glBindFramebuffer
#define glFramebufferTexture2D   p_glFramebufferTexture2D
#define glActiveTexture          p_glActiveTexture
#define glDrawElements           p_glDrawElements
#define glGenRenderbuffers       p_glGenRenderbuffers
#define glBindRenderbuffer       p_glBindRenderbuffer
#define glRenderbufferStorage    p_glRenderbufferStorage
#define glFramebufferRenderbuffer p_glFramebufferRenderbuffer
#define glCheckFramebufferStatus  p_glCheckFramebufferStatus
#define glGenerateMipmap          p_glGenerateMipmap

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::string readFile(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// load a JPEG into a tiling RGB8 texture (mipmapped); 0 on failure
static unsigned loadTextureRGB(const std::string& path)
{
    unsigned char* px = nullptr;
    int w = 0, h = 0;
    if (!loadJPEG(path.c_str(), &px, &w, &h)) {
        fprintf(stderr, "texture load failed: %s\n", path.c_str());
        return 0;
    }
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, px);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    delete[] px;
    printf("[assets] loaded %s (%dx%d)\n", path.c_str(), w, h);
    return t;
}

unsigned Renderer::compileProgram(const char* vertSrc, const char* fragSrc)
{
    auto compile = [&](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            fprintf(stderr, "shader error:\n%s\n", log);
        }
        return s;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragSrc);
    GLint vsOk = 0, fsOk = 0;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &vsOk);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &fsOk);
    if (!vsOk || !fsOk) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "link error:\n%s\n", log);
        return 0;
    }
    return p;
}

unsigned Renderer::makeColorTarget(int w, int h, unsigned* tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, *tex, 0);
    // depth for the scene pass
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        fprintf(stderr, "FBO incomplete\n");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fbo;
}

// ---------------------------------------------------------------------------
// camera: long-focus frustum at low optical height — the "Hokusai lens".
// A narrow FOV (22 deg) compresses depth so distant Fuji looms while the
// low eye point makes the foreground swell tower in frame.
// ---------------------------------------------------------------------------
static void matPerspective(float* m, float fovY, float aspect, float zn, float zf)
{
    const float t = 1.0f / tanf(fovY * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0] = t / aspect; m[5] = t;
    m[10] = (zf + zn) / (zn - zf); m[11] = -1.0f;
    m[14] = 2.0f * zf * zn / (zn - zf);
}
static void matLookAt(float* m, const float* eye, const float* at, const float* up)
{
    float f[3] = { at[0]-eye[0], at[1]-eye[1], at[2]-eye[2] };
    float fl = sqrtf(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    f[0]/=fl; f[1]/=fl; f[2]/=fl;
    float s[3] = { f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0] };
    float sl = sqrtf(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    s[0]/=sl; s[1]/=sl; s[2]/=sl;
    float u[3] = { s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0] };
    m[0]=s[0]; m[4]=s[1]; m[8] =s[2]; m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    m[1]=u[0]; m[5]=u[1]; m[9] =u[2]; m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    m[2]=-f[0];m[6]=-f[1];m[10]=-f[2];m[14]= (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
    m[3]=0;    m[7]=0;    m[11]=0;    m[15]=1;
}
static void matMul(float* o, const float* a, const float* b)
{
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a[k*4+r] * b[c*4+k];
            o[c*4+r] = s;
        }
}

void Renderer::buildCamera(float time, int targetW, int targetH)
{
    const float aspect = (float)targetW / (float)targetH;
    float proj[16], view[16];
    // far plane must reach the real Fuji terrain (~90 km out)
    matPerspective(proj, cfg_.fovDeg * 3.14159265f / 180.0f, aspect, 0.5f,
                   250000.0f);

    // Natural handheld ocean camera motion: subtle organic sway + wave-sync heave
    const float swayX = 18.0f * sinf(0.24f * time + 0.5f) + 12.0f * cosf(0.55f * time);
    const float swayY = 1.8f * sinf(0.38f * time + 1.2f) + 1.2f * cosf(0.81f * time);
    const float swayZ = 22.0f * cosf(0.18f * time);

    camPos_[0] = swayX;
    camPos_[1] = cfg_.camHeight + swayY;
    camPos_[2] = -350.0f + swayZ;               // Positioned in deep open sea facing Fuji

    // Bearing 289 deg faces Mt. Fuji from the Uraga Channel entrance
    const float brg = (289.0f + 0.45f * sinf(0.31f * time)) * 3.14159265f / 180.0f;
    const float dir[3] = { sinf(brg), 0.0f, -cosf(brg) };
    const float at[3]  = { camPos_[0] + dir[0] * 5000.0f,
                           camPos_[1] - 55.0f + 3.0f * cosf(0.42f * time),
                           camPos_[2] + dir[2] * 5000.0f };
    const float up[3]  = { 0.015f * sinf(0.27f * time), 1.0f, 0.0f };

    float f[3] = { at[0]-camPos_[0], at[1]-camPos_[1], at[2]-camPos_[2] };
    float fl = sqrtf(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    f[0]/=fl; f[1]/=fl; f[2]/=fl;
    camDir_[0]=f[0]; camDir_[1]=f[1]; camDir_[2]=f[2];
    camRight_[0] = f[1]*up[2]-f[2]*up[1];
    camRight_[1] = f[2]*up[0]-f[0]*up[2];
    camRight_[2] = f[0]*up[1]-f[1]*up[0];
    float sl = sqrtf(camRight_[0]*camRight_[0]+camRight_[1]*camRight_[1]
                     +camRight_[2]*camRight_[2]);
    camRight_[0]/=sl; camRight_[1]/=sl; camRight_[2]/=sl;
    camUp_[0] = camRight_[1]*f[2]-camRight_[2]*f[1];
    camUp_[1] = camRight_[2]*f[0]-camRight_[0]*f[2];
    camUp_[2] = camRight_[0]*f[1]-camRight_[1]*f[0];

    matLookAt(view, camPos_, at, up);
    matMul(viewProj_, proj, view);
}

// ---------------------------------------------------------------------------
// real Mount Fuji terrain mesh (Mapzen Terrarium DEM)
// ---------------------------------------------------------------------------
bool Renderer::loadTerrain()
{
    TerrainMesh mesh;
    // geographic origin = wave-patch center (35.145 N, 139.7 E);
    // mild height exaggeration keeps the real profile readable at ~90 km
    if (!loadFujiDEM((cfg_.assetsDir + "/fuji_dem.rgb").c_str(),
                     (cfg_.assetsDir + "/fuji_dem.meta").c_str(),
                     139.7f, 35.145f, /*heightExag=*/1.3f, /*decimate=*/4,
                     mesh)) {
        fprintf(stderr, "Fuji DEM not found (run assets/fetch_fuji_dem.sh)\n");
        return false;
    }
    terrainIdxCount_ = (int)mesh.idx.size();

    glGenVertexArrays(1, &vaoTerrain_);
    glBindVertexArray(vaoTerrain_);
    glGenBuffers(1, &vboTerrain_);
    glBindBuffer(GL_ARRAY_BUFFER, vboTerrain_);
    glBufferData(GL_ARRAY_BUFFER, mesh.verts.size() * sizeof(float),
                 mesh.verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (const void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (const void*)(4 * sizeof(float)));
    glGenBuffers(1, &iboTerrain_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTerrain_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 mesh.idx.size() * sizeof(unsigned), mesh.idx.data(),
                 GL_STATIC_DRAW);
    glBindVertexArray(0);
    printf("[terrain] real Fuji DEM mesh: %d x %d, %d indices\n",
           mesh.gridW, mesh.gridH, terrainIdxCount_);
    return true;
}

// ---------------------------------------------------------------------------
// updateBarrel — build the plunging-jet ribbon from the wave field.
//
// For each rib across the crest line we find the breaking crest point in
// the actual height field, then launch the lip on the ballistic plunging-
// jet trajectory used in breaking-wave theory (Cokelet & Longuet-Higgins):
// the sheet leaves the crest at about the wave phase speed and curls over
// under gravity. The result is a true overturning "C" surface, generated
// from the simulated wave — not a canned shape.
// ---------------------------------------------------------------------------
void Renderer::updateBarrel(const float* h, const float* foam, int n,
                            float domain, const float pdir[2],
                            float bankX, float bankZ, float phaseSpeed)
{
    const float q[2] = { -pdir[1], pdir[0] };        // crest-line direction
    const int nRibs = 71, nTau = 12;
    const int maxV = 80 * 12;
    std::vector<float> verts;
    verts.reserve(maxV * 5);
    std::vector<unsigned> idx;
    idx.reserve(79 * 11 * 6);

    auto sampleH = [&](float wx, float wz) {
        const float fx = (wx / domain + 0.5f) * (n - 1);
        const float fy = (wz / domain + 0.5f) * (n - 1);
        const int x0 = std::min(std::max((int)fx, 0), n - 2);
        const int y0 = std::min(std::max((int)fy, 0), n - 2);
        const float ax = fx - (float)x0, ay = fy - (float)y0;
        const float h00 = h[(size_t)y0 * n + x0], h10 = h[(size_t)y0 * n + x0 + 1];
        const float h01 = h[(size_t)(y0 + 1) * n + x0];
        const float h11 = h[(size_t)(y0 + 1) * n + x0 + 1];
        return (h00 * (1 - ax) + h10 * ax) * (1 - ay)
             + (h01 * (1 - ax) + h11 * ax) * ay;
    };
    auto sampleF = [&](float wx, float wz) {
        const int x = std::min(std::max((int)((wx / domain + 0.5f) * n), 0), n - 1);
        const int y = std::min(std::max((int)((wz / domain + 0.5f) * n), 0), n - 1);
        return foam[(size_t)y * n + x];
    };

    const float g = 9.81f;

    // first pass: crest candidates per rib
    struct Rib { float sq, sp, h; bool ok; };
    std::vector<Rib> ribs;
    for (int r = 0; r < nRibs; ++r) {
        const float sq = -700.0f + 20.0f * (float)r;
        float bestH = -1e9f, bestSp = 0.0f;
        for (float sp = -200.0f; sp <= 300.0f; sp += 8.0f) {
            const float wx = bankX + q[0] * sq + pdir[0] * sp;
            const float wz = bankZ + q[1] * sq + pdir[1] * sp;
            const float hv = sampleH(wx, wz);
            if (hv > bestH) { bestH = hv; bestSp = sp; }
        }
        const float wx = bankX + q[0] * sq + pdir[0] * bestSp;
        const float wz = bankZ + q[1] * sq + pdir[1] * bestSp;
        const bool ok = bestH > 3.0f && sampleF(wx, wz) > 0.3f;
        ribs.push_back({ sq, bestSp, bestH, ok });
    }
    // coherent crest line only: median s_p, reject outliers so no
    // stretched triangles span unrelated crest fragments
    std::vector<float> sps;
    for (auto& rb : ribs) if (rb.ok) sps.push_back(rb.sp);
    float med = 0.0f;
    if (!sps.empty()) {
        std::sort(sps.begin(), sps.end());
        med = sps[sps.size() / 2];
    }
    for (auto& rb : ribs)
        if (rb.ok && fabsf(rb.sp - med) > 35.0f) rb.ok = false;

    int nValid = 0;
    unsigned base = 0;
    float prevSp = 1e9f;
    bool prevOk = false;
    for (const auto& rb : ribs) {
        if (!rb.ok) { prevOk = false; continue; }
        const bool link = prevOk && fabsf(rb.sp - prevSp) < 35.0f;
        const float wx = bankX + q[0] * rb.sq + pdir[0] * rb.sp;
        const float wz = bankZ + q[1] * rb.sq + pdir[1] * rb.sp;

        // ballistic jet cross-section from the crest
        const float v0 = phaseSpeed * 1.1f;
        const float vy0 = 0.35f * sqrtf(g * rb.h);
        const float T = 0.85f;
        for (int k = 0; k < nTau; ++k) {
            const float tau = (float)k / (float)(nTau - 1);
            const float t = tau * T;
            const float px = wx + pdir[0] * (v0 * t);
            const float py = rb.h + vy0 * t - 0.5f * g * t * t;
            const float pz = wz + pdir[1] * (v0 * t);
            verts.insert(verts.end(), { px, py, pz, tau, rb.sq });
        }
        if (link) {
            const unsigned a0 = base - nTau, b0 = base;
            for (int k = 0; k < nTau - 1; ++k) {
                const unsigned a = a0 + k, b = a + 1, c = b0 + k, d = c + 1;
                idx.insert(idx.end(), { a, c, b, b, c, d });
            }
        }
        base += nTau;
        prevSp = rb.sp;
        prevOk = true;
        ++nValid;
    }

    barrelIdxCount_ = (int)idx.size();
    if (nValid == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, vboBarrel_);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    (GLsizeiptr)(verts.size() * sizeof(float)),
                    verts.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboBarrel_);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                    (GLsizeiptr)(idx.size() * sizeof(unsigned)), idx.data());
}

// ---------------------------------------------------------------------------
// spray droplet rendering: GL point sprites fed from a CUDA device buffer
// ---------------------------------------------------------------------------
bool Renderer::initSpray(int maxParticles)
{
    sprayMaxP_ = maxParticles;
    glGenVertexArrays(1, &vaoSpray_);
    glBindVertexArray(vaoSpray_);
    glGenBuffers(1, &vboSpray_);
    glBindBuffer(GL_ARRAY_BUFFER, vboSpray_);
    glBufferData(GL_ARRAY_BUFFER, (size_t)maxParticles * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          nullptr);
    // per-droplet diameter attribute (m)
    glGenBuffers(1, &vboSpraySize_);
    glBindBuffer(GL_ARRAY_BUFFER, vboSpraySize_);
    glBufferData(GL_ARRAY_BUFFER, (size_t)maxParticles * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), nullptr);
    glBindVertexArray(0);
    const cudaError_t e = cudaGraphicsGLRegisterBuffer(
        &resSpray_, vboSpray_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (e != cudaSuccess) {
        fprintf(stderr, "spray VBO interop failed: %s\n",
                cudaGetErrorString(e));
        return false;
    }
    const cudaError_t e2 = cudaGraphicsGLRegisterBuffer(
        &resSpraySize_, vboSpraySize_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (e2 != cudaSuccess) {
        fprintf(stderr, "spray size VBO interop failed: %s\n",
                cudaGetErrorString(e2));
        return false;
    }
    return true;
}

void Renderer::drawSpray(const float4* devPositions, const float* devSizes,
                         int count, float time)
{
    if (!resSpray_ || count <= 0) return;
    cudaGraphicsMapResources(1, &resSpray_, 0);
    void* p = nullptr;
    size_t sz = 0;
    cudaGraphicsResourceGetMappedPointer(&p, &sz, resSpray_);
    cudaMemcpyAsync(p, devPositions, (size_t)count * sizeof(float4),
                    cudaMemcpyDeviceToDevice, 0);
    cudaGraphicsUnmapResources(1, &resSpray_, 0);
    if (resSpraySize_ && devSizes != nullptr) {
        cudaGraphicsMapResources(1, &resSpraySize_, 0);
        void* ps = nullptr;
        size_t ssz = 0;
        cudaGraphicsResourceGetMappedPointer(&ps, &ssz, resSpraySize_);
        cudaMemcpyAsync(ps, devSizes, (size_t)count * sizeof(float),
                        cudaMemcpyDeviceToDevice, 0);
        cudaGraphicsUnmapResources(1, &resSpraySize_, 0);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glUseProgram(progSpray_);
    glUniformMatrix4fv(glGetUniformLocation(progSpray_, "uViewProj"), 1,
                       GL_FALSE, viewProj_);
    const float fovY = cfg_.fovDeg * 3.14159265f / 180.0f;
    glUniform1f(glGetUniformLocation(progSpray_, "uPointScale"),
                (float)cfg_.height / (2.0f * tanf(fovY * 0.5f)));
    glUniform1f(glGetUniformLocation(progSpray_, "uMaxPx"),
                (float)cfg_.height * 0.02f);
    glUniform3f(glGetUniformLocation(progSpray_, "uSunColor"),
                cfg_.sunColor[0], cfg_.sunColor[1], cfg_.sunColor[2]);
    glUniform3f(glGetUniformLocation(progSpray_, "uSunDir"),
                cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
    glUniform1f(glGetUniformLocation(progSpray_, "uDomain"), cfg_.domain);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texHeight_);
    glUniform1i(glGetUniformLocation(progSpray_, "uHeight"), 0);
    glBindVertexArray(vaoSpray_);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}
bool Renderer::initSeaFar()
{
    std::vector<float> verts;
    std::vector<unsigned> idx;
    const float H = cfg_.domain * 0.5f;     // patch half-width
    const float F = 120000.0f;              // horizon reach
    // four rectangles ringing the patch: N, S, W, E strips
    const float rects[4][4] = {
        { -F, -F,  F, -H },   // north strip (z: -F..-H)
        { -F,  H,  F,  F },   // south strip
        { -F, -H, -H,  H },   // west strip
        {  H, -H,  F,  H },   // east strip
    };
    unsigned base = 0;
    for (int r = 0; r < 4; ++r) {
        const int nx = 8, nz = 8;
        for (int j = 0; j <= nz; ++j)
            for (int i = 0; i <= nx; ++i) {
                const float x = rects[r][0] + (rects[r][2] - rects[r][0])
                              * (float)i / (float)nx;
                const float z = rects[r][1] + (rects[r][3] - rects[r][1])
                              * (float)j / (float)nz;
                verts.insert(verts.end(), { x, 0.0f, z, 0.0f });
            }
        for (int j = 0; j < nz; ++j)
            for (int i = 0; i < nx; ++i) {
                const unsigned a = base + j * (nx + 1) + i, b = a + 1;
                const unsigned c = a + (nx + 1), d = c + 1;
                idx.insert(idx.end(), { a, c, b, b, c, d });
            }
        base += (nx + 1) * (nz + 1);
    }
    seaFarIdxCount_ = (int)idx.size();

    glGenVertexArrays(1, &vaoSeaFar_);
    glBindVertexArray(vaoSeaFar_);
    glGenBuffers(1, &vboSeaFar_);
    glBindBuffer(GL_ARRAY_BUFFER, vboSeaFar_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float),
                 verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (const void*)(3 * sizeof(float)));
    glGenBuffers(1, &iboSeaFar_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboSeaFar_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned),
                 idx.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
    return true;
}

// ---------------------------------------------------------------------------
// EGL headless init
// ---------------------------------------------------------------------------
bool Renderer::init(const RenderConfig& cfg)
{
    cfg_ = cfg;

    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = getPlatform
        ? getPlatform(EGL_PLATFORM_DEVICE_EXT, nullptr, nullptr)
        : eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) { fprintf(stderr, "EGL: no display\n"); return false; }
    if (!eglInitialize(dpy, nullptr, nullptr)) { fprintf(stderr, "EGL init failed\n"); return false; }
    if (!eglBindAPI(EGL_OPENGL_API)) { fprintf(stderr, "EGL bind API failed\n"); return false; }

    const EGLint cfgAttrs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_NONE
    };
    EGLConfig ecfg;
    EGLint ncfg = 0;
    if (!eglChooseConfig(dpy, cfgAttrs, &ecfg, 1, &ncfg) || ncfg < 1) {
        fprintf(stderr, "EGL: no config\n"); return false;
    }
    const EGLint ctxAttrs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    EGLContext ctx = eglCreateContext(dpy, ecfg, EGL_NO_CONTEXT, ctxAttrs);
    if (ctx == EGL_NO_CONTEXT) { fprintf(stderr, "EGL: no GL 3.3 context\n"); return false; }
    const EGLint pbAttrs[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    EGLSurface surf = eglCreatePbufferSurface(dpy, ecfg, pbAttrs);
    if (!eglMakeCurrent(dpy, surf, surf, ctx)) {
        fprintf(stderr, "EGL: makeCurrent failed\n"); return false;
    }
    dpy_ = dpy; ctx_ = ctx; surf_ = surf;

    if (!loadGL()) { fprintf(stderr, "GL function load incomplete\n"); return false; }

    // --- shaders ---
    const std::string dir = cfg_.shaderDir + "/";
    const std::string bgV   = readFile(dir + "bg.vert");
    const std::string bgF   = readFile(dir + "bg.frag");
    const std::string ocV   = readFile(dir + "ocean.vert");
    const std::string ocF   = readFile(dir + "ocean.frag");
    const std::string abF   = readFile(dir + "aberration.frag");
    const std::string usF   = readFile(dir + "unsharp.frag");
    if (bgV.empty() || ocV.empty()) {
        fprintf(stderr, "shaders not found in %s\n", dir.c_str());
        return false;
    }
    progBg_     = compileProgram(bgV.c_str(), bgF.c_str());
    progOcean_  = compileProgram(ocV.c_str(), ocF.c_str());
    progAber_   = compileProgram(bgV.c_str(), abF.c_str());   // reuse fs-triangle vert
    progUnsharp_= compileProgram(bgV.c_str(), usF.c_str());
    const std::string teV = readFile(dir + "terrain.vert");
    const std::string teF = readFile(dir + "terrain.frag");
    progTerrain_ = compileProgram(teV.c_str(), teF.c_str());
    const std::string sfF = readFile(dir + "seafar.frag");
    progSeaFar_ = compileProgram(teV.c_str(), sfF.c_str());
    const std::string baV = readFile(dir + "barrel.vert");
    const std::string baF = readFile(dir + "barrel.frag");
    progBarrel_ = compileProgram(baV.c_str(), baF.c_str());
    const std::string spV = readFile(dir + "spray.vert");
    const std::string spF = readFile(dir + "spray.frag");
    progSpray_ = compileProgram(spV.c_str(), spF.c_str());
    if (!progBg_ || !progOcean_ || !progAber_ || !progUnsharp_ ||
        !progTerrain_ || !progSeaFar_ || !progBarrel_ || !progSpray_) {
        fprintf(stderr, "shader program setup failed; refusing to render a partial scene\n");
        return false;
    }

    // dynamic barrel ribbon buffers (rebuilt per frame)
    {
        const int maxV = 80 * 12, maxI = 79 * 11 * 6;
        glGenVertexArrays(1, &vaoBarrel_);
        glBindVertexArray(vaoBarrel_);
        glGenBuffers(1, &vboBarrel_);
        glBindBuffer(GL_ARRAY_BUFFER, vboBarrel_);
        glBufferData(GL_ARRAY_BUFFER, maxV * 5 * sizeof(float), nullptr,
                     GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              (const void*)(3 * sizeof(float)));
        glGenBuffers(1, &iboBarrel_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboBarrel_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, maxI * sizeof(unsigned),
                     nullptr, GL_DYNAMIC_DRAW);
        glBindVertexArray(0);
    }

    // --- fullscreen pass VAO (attribute-less) ---
    glGenVertexArrays(1, &vaoQuad_);

    // --- render targets ---
    fboW_ = cfg_.width; fboH_ = cfg_.height;
    fboScene_ = makeColorTarget(fboW_, fboH_, &texScene_);
    fboAber_  = makeColorTarget(fboW_, fboH_, &texAber_);
    fboTmp_   = makeColorTarget(fboW_, fboH_, &texTmp_);
    fboFinal_ = makeColorTarget(fboW_, fboH_, &texFinal_);

    // --- PBO for async readback (mapped into CUDA by postfx) ---
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_PACK_BUFFER,
                 (size_t)cfg_.width * cfg_.height * 4, nullptr, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    buildCamera(0.0f, cfg_.width, cfg_.height);

    // --- CC0 material maps (ambientCG, CC0 license) ---
    const std::string ad = cfg_.assetsDir + "/";
    texFoamAlbedo_  = loadTextureRGB(ad + "Foam001/Foam001_2K-JPG_Color.jpg");
    texFoamNormal_  = loadTextureRGB(ad + "Foam001/Foam001_2K-JPG_NormalGL.jpg");
    texFoamOpacity_ = loadTextureRGB(ad + "Foam003/Foam003_2K-JPG_Opacity.jpg");
    texFoamRough_   = loadTextureRGB(ad + "Foam001/Foam001_2K-JPG_Roughness.jpg");
    texRock_        = loadTextureRGB(ad + "Rock063/Rock063_2K-JPG_Color.jpg");
    loadTerrain();
    initSeaFar();
    return true;
}

// ---------------------------------------------------------------------------
// ocean mesh + textures
// ---------------------------------------------------------------------------
void Renderer::registerOceanTextures(int n)
{
    gridN_ = n;

    // mesh: n x n vertices, unit grid; indexed triangles
    std::vector<float> verts((size_t)n * n * 2);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            verts[((size_t)j * n + i) * 2 + 0] = (float)i / (float)(n - 1);
            verts[((size_t)j * n + i) * 2 + 1] = (float)j / (float)(n - 1);
        }
    std::vector<unsigned> idx;
    idx.reserve((size_t)(n - 1) * (n - 1) * 6);
    for (int j = 0; j < n - 1; ++j)
        for (int i = 0; i < n - 1; ++i) {
            const unsigned a = j * n + i, b = a + 1, c = a + n, d = c + 1;
            idx.insert(idx.end(), { a, c, b, b, c, d });
        }
    indexCount_ = (int)idx.size();

    glGenVertexArrays(1, &vaoGrid_);
    glBindVertexArray(vaoGrid_);
    glGenBuffers(1, &vboGrid_);
    glBindBuffer(GL_ARRAY_BUFFER, vboGrid_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glGenBuffers(1, &iboGrid_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboGrid_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned),
                 idx.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    auto makeFloatTex = [&](GLuint* t, GLint internal, GLenum format) {
        glGenTextures(1, t);
        glBindTexture(GL_TEXTURE_2D, *t);
        glTexImage2D(GL_TEXTURE_2D, 0, internal, n, n, 0, format, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    makeFloatTex(&texHeight_, GL_R32F, GL_RED);
    makeFloatTex(&texDisp_,   GL_RG32F, GL_RG);
    makeFloatTex(&texFoam_,   GL_R32F, GL_RED);
    makeFloatTex(&texDepth_,  GL_R32F, GL_RED);
    makeFloatTex(&texGain_,   GL_R32F, GL_RED);

    // CUDA interop for the per-frame animated textures
    cudaError_t e1 = cudaGraphicsGLRegisterImage(&resHeight_, texHeight_,
                                                 GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsWriteDiscard);
    cudaError_t e2 = cudaGraphicsGLRegisterImage(&resDisp_, texDisp_,
                                                 GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsWriteDiscard);
    cudaError_t e3 = cudaGraphicsGLRegisterImage(&resFoam_, texFoam_,
                                                 GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsWriteDiscard);
    interopOk_ = (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess);
    if (!interopOk_)
        fprintf(stderr, "warning: CUDA-GL image interop unavailable, host fallback\n");
}

void Renderer::uploadOcean(const float* h, const float2* disp, const float* foam,
                           const float* depth, const float* gain,
                           cudaStream_t stream)
{
    const int n = gridN_;
    const size_t cells = (size_t)n * n;

    if (interopOk_) {
        cudaGraphicsResource* res[3] = { resHeight_, resDisp_, resFoam_ };
        cudaError_t em = cudaGraphicsMapResources(3, res, stream);
        cudaArray_t arr = nullptr;
        cudaError_t e1 = cudaGraphicsSubResourceGetMappedArray(&arr, resHeight_, 0, 0);
        cudaError_t c1 = cudaMemcpy2DToArrayAsync(arr, 0, 0, h, n * sizeof(float),
                                 n * sizeof(float), n,
                                 cudaMemcpyDeviceToDevice, stream);
        cudaError_t e2 = cudaGraphicsSubResourceGetMappedArray(&arr, resDisp_, 0, 0);
        cudaError_t c2 = cudaMemcpy2DToArrayAsync(arr, 0, 0, disp, n * sizeof(float2),
                                 n * sizeof(float2), n,
                                 cudaMemcpyDeviceToDevice, stream);
        cudaError_t e3 = cudaGraphicsSubResourceGetMappedArray(&arr, resFoam_, 0, 0);
        cudaError_t c3 = cudaMemcpy2DToArrayAsync(arr, 0, 0, foam, n * sizeof(float),
                                 n * sizeof(float), n,
                                 cudaMemcpyDeviceToDevice, stream);
        cudaError_t eu = cudaGraphicsUnmapResources(3, res, stream);
        static bool reported = false;
        if (!reported) {
            reported = true;
            const char* names[] = { "map", "getH", "cpyH", "getD", "cpyD",
                                    "getF", "cpyF", "unmap" };
            const cudaError_t errs[] = { em, e1, c1, e2, c2, e3, c3, eu };
            for (int k = 0; k < 8; ++k)
                if (errs[k] != cudaSuccess)
                    fprintf(stderr, "interop %s: %s\n", names[k],
                            cudaGetErrorString(errs[k]));
            fprintf(stderr, "interop upload check done\n");
            // GL-side texture content probe
            std::vector<float> gbuf((size_t)n * n);
            glBindTexture(GL_TEXTURE_2D, texHeight_);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, gbuf.data());
            float mn = 1e30f, mx = -1e30f;
            for (float v : gbuf) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
            fprintf(stderr, "GL height texture min/max = %.3f / %.3f (err=0x%x)\n",
                    mn, mx, glGetError());
        }
    } else {
        // host fallback path
        std::vector<float> tmp(cells * 2);
        cudaMemcpyAsync(tmp.data(), h, cells * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        glBindTexture(GL_TEXTURE_2D, texHeight_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, tmp.data());
        cudaMemcpy(tmp.data(), disp, cells * sizeof(float2), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, texDisp_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RG, GL_FLOAT, tmp.data());
        cudaMemcpy(tmp.data(), foam, cells * sizeof(float), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, texFoam_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, tmp.data());
    }

    // static maps: upload once (cheap guard via a static flag per call site)
    static bool staticDone = false;
    if (!staticDone) {
        std::vector<float> tmp(cells);
        cudaMemcpy(tmp.data(), depth, cells * sizeof(float), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, texDepth_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, tmp.data());
        cudaMemcpy(tmp.data(), gain, cells * sizeof(float), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, texGain_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, n, GL_RED, GL_FLOAT, tmp.data());
        staticDone = true;
    }
}

// ---------------------------------------------------------------------------
// frame rendering
// ---------------------------------------------------------------------------
static void bindTex(int unit, GLuint tex, GLint loc, int idx)
{
    glActiveTexture(GL_TEXTURE0 + unit);   // unit is an INDEX (0,1,2,...)
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(loc, idx);
}

void Renderer::renderFrame(float time, float tide, const float4* sprayPos,
                           const float* spraySizes, int sprayCount, int w, int h)
{
    buildCamera(time, w, h);
    glViewport(0, 0, w, h);

    // Ensure PBO is correctly sized for the current viewport (initialized at maxH)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    // No resize needed if we initialized with maxH, but we must use correct size for readback

    // ---- pass 1: background + ocean into the scene FBO ----
    glBindFramebuffer(GL_FRAMEBUFFER, fboScene_);
    glClearColor(0.05f, 0.1f, 0.16f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glUseProgram(progBg_);
    glUniform1f(glGetUniformLocation(progBg_, "uTime"), time);
    glUniform1f(glGetUniformLocation(progBg_, "uFovY"),
                cfg_.fovDeg * 3.14159265f / 180.0f);
    glUniform1f(glGetUniformLocation(progBg_, "uAspect"), (float)w / (float)h);
    glUniform1f(glGetUniformLocation(progBg_, "uCloud"), cfg_.cloudAmount);
    glUniform3f(glGetUniformLocation(progBg_, "uSunDir"),
                cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
    glUniform3f(glGetUniformLocation(progBg_, "uSunColor"),
                cfg_.sunColor[0], cfg_.sunColor[1], cfg_.sunColor[2]);
    // camera basis matching buildCamera()
    glUniform3f(glGetUniformLocation(progBg_, "uCamDir"),
                camDir_[0], camDir_[1], camDir_[2]);
    glUniform3f(glGetUniformLocation(progBg_, "uCamRight"),
                camRight_[0], camRight_[1], camRight_[2]);
    glUniform3f(glGetUniformLocation(progBg_, "uCamUp"),
                camUp_[0], camUp_[1], camUp_[2]);
    bindTex(0, texRock_, glGetUniformLocation(progBg_, "uRockTex"), 0);
    glUniform1i(glGetUniformLocation(progBg_, "uHasRock"),
                texRock_ ? 1 : 0);
    glBindVertexArray(vaoQuad_);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // ---- real Fuji terrain (DEM mesh) between sky and ocean ----
    if (vaoTerrain_) {
        glEnable(GL_DEPTH_TEST);
        glUseProgram(progTerrain_);
        glUniformMatrix4fv(glGetUniformLocation(progTerrain_, "uViewProj"),
                           1, GL_FALSE, viewProj_);
        glUniform3f(glGetUniformLocation(progTerrain_, "uCamPos"),
                    camPos_[0], camPos_[1], camPos_[2]);
        glUniform3f(glGetUniformLocation(progTerrain_, "uSunDir"),
                    cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
        // horizon fog tone: identical to the sea/sky below-horizon fog
        // (0.42,0.46,0.52 * sun * 0.7875) so distant lowlands, far sea and
        // sky melt into ONE continuous tone instead of a bright blue band
        const float* sc = cfg_.sunColor;
        const float hzR = sc[0] * 0.42f * 0.7875f;
        const float hzG = sc[1] * 0.46f * 0.7875f;
        const float hzB = sc[2] * 0.52f * 0.7875f;
        glUniform3f(glGetUniformLocation(progTerrain_, "uHorizonCol"),
                    hzR, hzG, hzB);
        glUniform1f(glGetUniformLocation(progTerrain_, "uTide"), 0.0f);
        bindTex(0, texRock_, glGetUniformLocation(progTerrain_, "uRockTex"), 0);
        glBindVertexArray(vaoTerrain_);
        glDrawElements(GL_TRIANGLES, terrainIdxCount_, GL_UNSIGNED_INT,
                       nullptr);
        glBindVertexArray(0);
    }

    // ---- far sea plane: bay water out to the horizon (no floating coast)
    if (vaoSeaFar_) {
        glEnable(GL_DEPTH_TEST);
        glUseProgram(progSeaFar_);
        glUniformMatrix4fv(glGetUniformLocation(progSeaFar_, "uViewProj"),
                           1, GL_FALSE, viewProj_);
        glUniform3f(glGetUniformLocation(progSeaFar_, "uCamPos"),
                    camPos_[0], camPos_[1], camPos_[2]);
        glUniform3f(glGetUniformLocation(progSeaFar_, "uSunDir"),
                    cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
        glUniform3f(glGetUniformLocation(progSeaFar_, "uSunColor"),
                    cfg_.sunColor[0], cfg_.sunColor[1], cfg_.sunColor[2]);
        glUniform1f(glGetUniformLocation(progSeaFar_, "uTide"), tide);
        glUniform1f(glGetUniformLocation(progSeaFar_, "uTime"), time);
        glBindVertexArray(vaoSeaFar_);
        glDrawElements(GL_TRIANGLES, seaFarIdxCount_, GL_UNSIGNED_INT,
                       nullptr);
        glBindVertexArray(0);
    }

    glEnable(GL_DEPTH_TEST);
    glUseProgram(progOcean_);
    glUniformMatrix4fv(glGetUniformLocation(progOcean_, "uViewProj"), 1,
                       GL_FALSE, viewProj_);
    glUniform1f(glGetUniformLocation(progOcean_, "uDomain"), cfg_.domain);
    glUniform1f(glGetUniformLocation(progOcean_, "uLambda"), cfg_.lambda);
    glUniform3f(glGetUniformLocation(progOcean_, "uCamPos"),
                camPos_[0], camPos_[1], camPos_[2]);
    glUniform3f(glGetUniformLocation(progOcean_, "uSunDir"),
                cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
    glUniform3f(glGetUniformLocation(progOcean_, "uSunColor"),
                cfg_.sunColor[0], cfg_.sunColor[1], cfg_.sunColor[2]);
    glUniform1f(glGetUniformLocation(progOcean_, "uTime"), time);
    glUniform1f(glGetUniformLocation(progOcean_, "uDomain"), cfg_.domain);
    glUniform1f(glGetUniformLocation(progOcean_, "uTide"), tide);
    glUniform2f(glGetUniformLocation(progOcean_, "uPropDir"),
                cfg_.propDir[0], cfg_.propDir[1]);
    bindTex(0, texHeight_, glGetUniformLocation(progOcean_, "uHeight"), 0);
    bindTex(1, texDisp_,   glGetUniformLocation(progOcean_, "uDisp"), 1);
    bindTex(2, texFoam_,   glGetUniformLocation(progOcean_, "uFoam"), 2);
    bindTex(3, texDepth_,  glGetUniformLocation(progOcean_, "uDepth"), 3);
    bindTex(4, texFoamAlbedo_,  glGetUniformLocation(progOcean_, "uFoamAlbedo"), 4);
    bindTex(5, texFoamNormal_,  glGetUniformLocation(progOcean_, "uFoamNormal"), 5);
    bindTex(6, texFoamOpacity_, glGetUniformLocation(progOcean_, "uFoamOpacity"), 6);
    bindTex(7, texFoamRough_,   glGetUniformLocation(progOcean_, "uFoamRough"), 7);
    glUniform1i(glGetUniformLocation(progOcean_, "uHasFoamTex"),
                (texFoamAlbedo_ && texFoamNormal_ && texFoamOpacity_) ? 1 : 0);
    glBindVertexArray(vaoGrid_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    // ---- barrel lip ribbon (disabled to prevent procedural geometry artifacts) ----
    if (false && barrelIdxCount_ > 0) {
        glUseProgram(progBarrel_);
        glUniformMatrix4fv(glGetUniformLocation(progBarrel_, "uViewProj"),
                           1, GL_FALSE, viewProj_);
        glUniform3f(glGetUniformLocation(progBarrel_, "uCamPos"),
                    camPos_[0], camPos_[1], camPos_[2]);
        glUniform3f(glGetUniformLocation(progBarrel_, "uSunDir"),
                    cfg_.sunDir[0], cfg_.sunDir[1], cfg_.sunDir[2]);
        glUniform3f(glGetUniformLocation(progBarrel_, "uSunColor"),
                    cfg_.sunColor[0], cfg_.sunColor[1], cfg_.sunColor[2]);
        glUniform1f(glGetUniformLocation(progBarrel_, "uTime"), time);
        glBindVertexArray(vaoBarrel_);
        glDrawElements(GL_TRIANGLES, barrelIdxCount_, GL_UNSIGNED_INT,
                       nullptr);
        glBindVertexArray(0);
    }

    // ---- spray droplets (point sprites, alpha-blended over the water) ----
    if (sprayCount > 0 && sprayPos != nullptr)
        drawSpray(sprayPos, spraySizes, sprayCount, time);

    // ---- pass 2: spherical aberration (both pipelines) ----
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, fboAber_);
    glViewport(0, 0, w, h);
    glUseProgram(progAber_);
    glUniform1f(glGetUniformLocation(progAber_, "uAmount"), cfg_.aberration);
    glUniform2f(glGetUniformLocation(progAber_, "uTexel"),
                1.0f / (float)fboW_, 1.0f / (float)fboH_);
    glUniform2f(glGetUniformLocation(progAber_, "uMaxUv"),
                (float)w / (float)fboW_, (float)h / (float)fboH_);
    bindTex(0, texScene_, glGetUniformLocation(progAber_, "uScene"), 0);
    glBindVertexArray(vaoQuad_);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Both pipelines sharpen the SAME aberration output afterwards in CUDA
    // (traditional: global-memory taps; complementary: warp-shuffle taps),
    // so the rendered input to the A/B post-fx is identical by construction.
    // THE TRADITIONAL GL PIPELINE (for control) performs unsharp masking here:
    {
        glUseProgram(progUnsharp_);
        glUniform2f(glGetUniformLocation(progUnsharp_, "uMaxUv"),
                    (float)w / (float)fboW_, (float)h / (float)fboH_);
        // Horizontal pass
        glBindFramebuffer(GL_FRAMEBUFFER, fboTmp_);
        glViewport(0, 0, w, h);
        glUniform1i(glGetUniformLocation(progUnsharp_, "uCombine"), 0);
        glUniform2f(glGetUniformLocation(progUnsharp_, "uDir"), 1.0f / (float)fboW_, 0.0f);
        bindTex(0, texAber_, glGetUniformLocation(progUnsharp_, "uImage"), 0);
        glBindVertexArray(vaoQuad_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // Vertical pass + Combine (rendered to fboFinal_ which is then read back)
        glBindFramebuffer(GL_FRAMEBUFFER, fboFinal_);
        glViewport(0, 0, w, h);
        glUniform1i(glGetUniformLocation(progUnsharp_, "uCombine"), 1);
        glUniform1f(glGetUniformLocation(progUnsharp_, "uAmount"), 1.2f);
        glUniform2f(glGetUniformLocation(progUnsharp_, "uDir"), 0.0f, 1.0f / (float)fboH_);
        bindTex(0, texTmp_, glGetUniformLocation(progUnsharp_, "uImage"), 0);
        bindTex(1, texAber_, glGetUniformLocation(progUnsharp_, "uOrig"), 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // ---- async readback into the PBO ----
    // For the traditional pipeline (idx 0 and 1 in main), we want the sharpened fboFinal_
    // For the complementary pipeline (idx 2 and 3), we want the raw aberration fboAber_ 
    // and CUDA will handle sharpening.
    // However, we are rendering one pipeline at a time now.
    // We can just read from the last written FBO.
    // Actually, we need to know if we are doing traditional or complementary.
    // Let's just always read from fboAber_ for now, BUT if we want to support
    // the traditional GL pipeline results, we'd read from fboFinal_.
    // In main.cpp, we use PostFx::beginFrame(0 or 1).
    // If correctionMode == 0 (traditional), CUDA will re-do the sharpening.
    // So reading from fboAber_ is consistent.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboAber_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::shutdown()
{
    if (resHeight_) { cudaGraphicsUnregisterResource(resHeight_); resHeight_ = nullptr; }
    if (resDisp_)   { cudaGraphicsUnregisterResource(resDisp_);   resDisp_ = nullptr; }
    if (resFoam_)   { cudaGraphicsUnregisterResource(resFoam_);   resFoam_ = nullptr; }
    if (dpy_) {
        eglMakeCurrent((EGLDisplay)dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext((EGLDisplay)dpy_, (EGLContext)ctx_);
        eglDestroySurface((EGLDisplay)dpy_, (EGLSurface)surf_);
        eglTerminate((EGLDisplay)dpy_);
        dpy_ = nullptr;
    }
}
