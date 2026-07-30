// ============================================================================
// bathymetry.cpp — synthetic Uraga Channel + NetCDF-3 classic reader/writer
// ============================================================================
#include "bathymetry.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Synthetic bathymetry: mouth of Tokyo Bay (Uraga Suido / Sagami-nada side)
// Domain: 0.36 deg x 0.36 deg centered on 35.2 N, 139.7 E (~40 km square).
//
// Features modeled:
//   - Uraga fairway: a curved deep channel (35-55 m) running S->N into the
//     bay, flanked by the Miura (west) and Boso (east) shallow banks (5-15 m)
//   - Sagami-nada continental slope: depth drops past 100 m toward the
//     southern (open Pacific) edge — this is the deep-water wave source
//   - small-scale sandbank undulations on the shelf
// ---------------------------------------------------------------------------
static float smoothstepf(float a, float b, float x)
{
    float t = (x - a) / (b - a);
    t = std::min(1.0f, std::max(0.0f, t));
    return t * t * (3.0f - 2.0f * t);
}

// tiny deterministic value noise
static float vnoise(float x, float y)
{
    const float v = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return v - floorf(v);
}

Bathymetry generateSyntheticUraga(int n)
{
    Bathymetry b;
    b.width = b.height = n;
    b.latMin = 35.2f - 0.18f;  b.latMax = 35.2f + 0.18f;
    b.lonMin = 139.7f - 0.18f; b.lonMax = 139.7f + 0.18f;
    b.elevation.resize((size_t)n * n);

    for (int iy = 0; iy < n; ++iy) {
        const float y = (float)iy / (float)(n - 1);      // 0=south, 1=north
        for (int ix = 0; ix < n; ++ix) {
            const float x = (float)ix / (float)(n - 1);  // 0=west, 1=east

            // Meandering fairway centerline (Uraga channel, S->N).
            const float xc = 0.52f + 0.10f * sinf(3.14159f * (1.6f * y + 0.15f));
            const float dChan = fabsf(x - xc);
            const float wChan = 0.055f + 0.02f * y;      // half-width widens N

            // Base shelf depth: shallow banks (8 m) blending into the
            // channel (45 m) by lateral distance from the centerline.
            float depth = 8.0f + 37.0f * (1.0f - smoothstepf(wChan * 0.6f,
                                                             wChan * 2.2f, dChan));
            // Open-Pacific slope: south of the bay mouth the floor falls
            // away into Sagami-nada (> 100 m) — deep-water incoming swell.
            depth += 95.0f * (1.0f - smoothstepf(0.02f, 0.30f, y));
            // Secondary Boso-side basin east of the channel.
            depth += 12.0f * smoothstepf(0.75f, 0.95f, x) * smoothstepf(0.3f, 0.6f, y);
            // Offshore breaking bank right in front of the viewpoint
            // (~300 m ahead of the camera). ASYMMETRIC profile: gentle
            // seaward slope but a steep shoreward edge, so the swell rear
            // rises slowly while the crest accelerates over the lip and
            // pitches forward — the physical plunging-breaker mechanism
            // that curls the crest into a C shape. Elongated ACROSS the
            // swell direction so the curling crest line runs wide.
            {
                const float bx = 0.4938f, by = 0.3270f;
                const float sx = (x - bx) / 0.030f;
                // waves travel roughly NNE (+lat): steep shoreward edge
                const float sy = (y - by) / ((y > by) ? 0.008f : 0.022f);
                const float bank = expf(-0.5f * (sx * sx + sy * sy));
                depth = depth * (1.0f - bank) + 3.0f * bank;
            }
            // Beach ramp along the south edge of the wave patch: depth
            // shoals toward a shoreline right next to the lens, so the
            // swell sweeps in and curls over the near shore (C-shaped
            // shorebreak) instead of rolling past as smooth hills.
            {
                const float y0 = 0.337f;   // ramp start  (z = +400 m)
                const float y1 = 0.316f;   // shoreline   (z = +1250 m)
                float s = (y0 - y) / (y0 - y1);
                s = fminf(fmaxf(s, 0.0f), 1.0f);
                s = s * s * (3.0f - 2.0f * s);
                depth = depth * (1.0f - s) + 1.5f * s;
            }
            // Sandbank ripples.
            depth += 2.5f * (vnoise(x * 9.0f, y * 9.0f) - 0.5f);
            depth = std::max(2.0f, depth);

            b.elevation[(size_t)iy * n + ix] = -depth;   // GEBCO: negative
        }
    }
    return b;
}

// ---------------------------------------------------------------------------
// NetCDF-3 classic format (big-endian). Minimal but standards-compliant
// subset: dims lat/lon, vars lat/lon/elevation, NC_FLOAT / NC_SHORT with
// optional scale_factor / add_offset attributes on the data variable.
// ---------------------------------------------------------------------------
namespace nc3 {

static inline void put32(FILE* f, unsigned v)
{
    unsigned char b[4] = { (unsigned char)(v >> 24), (unsigned char)(v >> 16),
                           (unsigned char)(v >> 8),  (unsigned char)v };
    fwrite(b, 1, 4, f);
}
static inline unsigned get32(const unsigned char*& p)
{
    unsigned v = ((unsigned)p[0] << 24) | ((unsigned)p[1] << 16)
               | ((unsigned)p[2] << 8) | (unsigned)p[3];
    p += 4;
    return v;
}
static void putName(FILE* f, const char* s)
{
    const size_t len = strlen(s);
    put32(f, (unsigned)len);
    fwrite(s, 1, len, f);
    const size_t pad = (4 - (len & 3)) & 3;
    for (size_t i = 0; i < pad; ++i) fputc(0, f);
}
static void getName(const unsigned char*& p, char* out, size_t cap)
{
    const unsigned len = get32(p);
    const size_t n = std::min((size_t)len, cap - 1);
    memcpy(out, p, n);
    out[n] = 0;
    p += len + ((4 - (len & 3)) & 3);
}

} // namespace nc3

bool writeNetCDF3(const char* path, const Bathymetry& b)
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    const unsigned nLat = (unsigned)b.height, nLon = (unsigned)b.width;
    const unsigned latBytes = nLat * 4, lonBytes = nLon * 4;
    const unsigned elevBytes = nLat * nLon * 4;

    // Header size: magic(4) numrecs(4) dimlist(8 + 2*(4+padname+4))
    //              gatt ABSENT(8) varlist(8 + 3 vars)
    // var header: name + 4(ndims) + 4*ndims(dimids) + 8(vatt ABSENT)
    //             + 4(type) + 4(vsize) + 4(begin)
    const unsigned nameLat = 4 + 4, nameLon = 4 + 4, nameEl = 4 + 12; // "elevation"=9->12
    const unsigned dimList = 8 + (nameLat + 4) + (nameLon + 4);
    const unsigned varLat  = nameLat + 4 + 4 + 8 + 12;
    const unsigned varLon  = nameLon + 4 + 4 + 8 + 12;
    const unsigned varEl   = nameEl + 4 + 8 + 8 + 12;
    const unsigned header  = 8 + dimList + 8 + 8 + varLat + varLon + varEl;

    const unsigned beginLat  = header;
    const unsigned beginLon  = beginLat + latBytes;
    const unsigned beginElev = beginLon + lonBytes;

    fwrite("CDF\001", 1, 4, f);          // classic format
    nc3::put32(f, 0);                    // numrecs (fixed-size only)

    nc3::put32(f, 0x0A);                 // NC_DIMENSION
    nc3::put32(f, 2);
    nc3::putName(f, "lat"); nc3::put32(f, nLat);
    nc3::putName(f, "lon"); nc3::put32(f, nLon);

    nc3::put32(f, 0); nc3::put32(f, 0);  // global attrs: ABSENT

    nc3::put32(f, 0x0B);                 // NC_VARIABLE
    nc3::put32(f, 3);
    // var: lat
    nc3::putName(f, "lat");
    nc3::put32(f, 1); nc3::put32(f, 0);
    nc3::put32(f, 0); nc3::put32(f, 0);
    nc3::put32(f, 5); nc3::put32(f, latBytes); nc3::put32(f, beginLat);
    // var: lon
    nc3::putName(f, "lon");
    nc3::put32(f, 1); nc3::put32(f, 1);
    nc3::put32(f, 0); nc3::put32(f, 0);
    nc3::put32(f, 5); nc3::put32(f, lonBytes); nc3::put32(f, beginLon);
    // var: elevation
    nc3::putName(f, "elevation");
    nc3::put32(f, 2); nc3::put32(f, 0); nc3::put32(f, 1);
    nc3::put32(f, 0); nc3::put32(f, 0);
    nc3::put32(f, 5); nc3::put32(f, elevBytes); nc3::put32(f, beginElev);

    // data (big-endian floats)
    std::vector<float> lat(nLat), lon(nLon);
    for (unsigned i = 0; i < nLat; ++i)
        lat[i] = b.latMin + (b.latMax - b.latMin) * (float)i / (float)(nLat - 1);
    for (unsigned i = 0; i < nLon; ++i)
        lon[i] = b.lonMin + (b.lonMax - b.lonMin) * (float)i / (float)(nLon - 1);

    unsigned char buf[4];
    auto writeFloats = [&](const float* p, size_t cnt) {
        for (size_t i = 0; i < cnt; ++i) {
            unsigned u;
            memcpy(&u, &p[i], 4);
            buf[0] = (unsigned char)(u >> 24); buf[1] = (unsigned char)(u >> 16);
            buf[2] = (unsigned char)(u >> 8);  buf[3] = (unsigned char)u;
            fwrite(buf, 1, 4, f);
        }
    };
    writeFloats(lat.data(), nLat);
    writeFloats(lon.data(), nLon);
    writeFloats(b.elevation.data(), (size_t)nLat * nLon);

    fclose(f);
    return true;
}

bool loadNetCDF3(const char* path, Bathymetry& out)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> file((size_t)sz);
    if (fread(file.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);

    const unsigned char* p = file.data();
    if (sz < 8 || memcmp(p, "CDF", 3) != 0 || p[3] > 2) return false; // classic/64-bit offset
    const bool offset64 = (p[3] == 2);
    p += 4;
    nc3::get32(p);                                     // numrecs

    // --- dimensions ---
    unsigned tag = nc3::get32(p);
    unsigned nDims = 0;
    char dimNames[16][64];
    unsigned dimSizes[16];
    if (tag == 0x0A) {
        nDims = nc3::get32(p);
        for (unsigned i = 0; i < nDims && i < 16; ++i) {
            nc3::getName(p, dimNames[i], 64);
            dimSizes[i] = nc3::get32(p);
        }
    } else {
        nc3::get32(p);                                 // ABSENT count
    }

    // --- global attributes (skipped generically) ---
    tag = nc3::get32(p);
    if (tag == 0x0C) {
        const unsigned nAtt = nc3::get32(p);
        for (unsigned i = 0; i < nAtt; ++i) {
            char nm[64]; nc3::getName(p, nm, 64);
            const unsigned type = nc3::get32(p);
            const unsigned ne   = nc3::get32(p);
            static const unsigned tsize[7] = {0,1,1,2,4,4,8};
            const unsigned bytes = ne * tsize[type & 7];
            p += bytes + ((4 - (bytes & 3)) & 3);
        }
    } else {
        nc3::get32(p);
    }

    // --- variables ---
    tag = nc3::get32(p);
    if (tag != 0x0B) return false;
    const unsigned nVars = nc3::get32(p);

    int latDim = -1, lonDim = -1;
    int idxLat = -1, idxLon = -1;
    for (unsigned i = 0; i < nDims && i < 16; ++i) {
        if (strncmp(dimNames[i], "lat", 3) == 0 || strncmp(dimNames[i], "y", 1) == 0)
            { latDim = (int)i; idxLat = (int)i; }
        if (strncmp(dimNames[i], "lon", 3) == 0 || strncmp(dimNames[i], "x", 1) == 0)
            { lonDim = (int)i; idxLon = (int)i; }
    }
    if (latDim < 0 || lonDim < 0 || nDims < 2) return false;
    if (latDim > lonDim) { /* GEBCO stores [lat][lon]; keep as-is */ }

    out.height = (int)dimSizes[latDim];
    out.width  = (int)dimSizes[lonDim];
    std::vector<float> latv(out.height), lonv(out.width);
    bool haveElev = false;

    for (unsigned v = 0; v < nVars; ++v) {
        char nm[64]; nc3::getName(p, nm, 64);
        const unsigned nd = nc3::get32(p);
        unsigned dimIds[8];
        for (unsigned d = 0; d < nd && d < 8; ++d) dimIds[d] = nc3::get32(p);

        // variable attributes: harvest scale_factor / add_offset
        float scale = 1.0f, offset = 0.0f;
        tag = nc3::get32(p);
        if (tag == 0x0C) {
            const unsigned nAtt = nc3::get32(p);
            for (unsigned a = 0; a < nAtt; ++a) {
                char anm[64]; nc3::getName(p, anm, 64);
                const unsigned type = nc3::get32(p);
                const unsigned ne   = nc3::get32(p);
                static const unsigned tsize[7] = {0,1,1,2,4,4,8};
                const unsigned bytes = ne * tsize[type & 7];
                if ((strcmp(anm, "scale_factor") == 0 || strcmp(anm, "add_offset") == 0)
                    && type >= 5 && ne >= 1) {
                    double val = 0.0;
                    if (type == 5) {                     // NC_FLOAT
                        unsigned u = ((unsigned)p[0] << 24) | ((unsigned)p[1] << 16)
                                   | ((unsigned)p[2] << 8) | (unsigned)p[3];
                        float t2; memcpy(&t2, &u, 4); val = t2;
                    } else if (type == 6) {              // NC_DOUBLE
                        unsigned long long u = 0;
                        for (int b8 = 0; b8 < 8; ++b8) u = (u << 8) | p[b8];
                        memcpy(&val, &u, 8);
                    }
                    if (strcmp(anm, "scale_factor") == 0) scale = (float)val;
                    else                                  offset = (float)val;
                }
                p += bytes + ((4 - (bytes & 3)) & 3);
            }
        } else {
            nc3::get32(p);
        }

        const unsigned type  = nc3::get32(p);
        nc3::get32(p);                                 // vsize
        unsigned long long begin = nc3::get32(p);
        if (offset64) { p -= 4; unsigned hi = nc3::get32(p), lo = nc3::get32(p);
                        begin = ((unsigned long long)hi << 32) | lo; }

        const bool isLat = (nd == 1 && (int)dimIds[0] == idxLat);
        const bool isLon = (nd == 1 && (int)dimIds[0] == idxLon);
        const bool isElev = (nd == 2)
            && (strcmp(nm, "elevation") == 0 || strcmp(nm, "z") == 0
                || strcmp(nm, "Band1") == 0 || strcmp(nm, "height") == 0);
        if (!isLat && !isLon && !isElev) continue;

        const unsigned char* dp = file.data() + begin;
        const size_t cnt = isElev ? (size_t)out.width * out.height
                                  : (isLat ? latv.size() : lonv.size());
        std::vector<float>* dst = isLat ? &latv : isLon ? &lonv : nullptr;
        if (isElev) { out.elevation.resize(cnt); dst = &out.elevation; haveElev = true; }

        for (size_t i = 0; i < cnt; ++i) {
            float val = 0.f;
            if (type == 5) {                             // NC_FLOAT
                unsigned u = nc3::get32(dp); memcpy(&val, &u, 4);
            } else if (type == 3) {                      // NC_SHORT
                short sv = (short)((dp[0] << 8) | dp[1]); dp += 2;
                val = (float)sv;
            } else if (type == 4) {                      // NC_INT
                val = (float)(int)nc3::get32(dp);
            } else if (type == 6) {                      // NC_DOUBLE
                unsigned hi = nc3::get32(dp), lo = nc3::get32(dp);
                unsigned long long u = ((unsigned long long)hi << 32) | lo;
                double dv; memcpy(&dv, &u, 8); val = (float)dv;
            } else { haveElev = false; break; }
            (*dst)[i] = isElev ? val * scale + offset : val;
        }
        if (isElev) { /* scale/offset applied above */ }
        (void)scale; (void)offset;
    }

    if (!haveElev) return false;
    out.latMin = latv.front(); out.latMax = latv.back();
    out.lonMin = lonv.front(); out.lonMax = lonv.back();
    return true;
}
