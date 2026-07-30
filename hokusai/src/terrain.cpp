// ============================================================================
// terrain.cpp — Terrarium DEM -> geographically anchored world-space mesh
//
// The DEM spans the full view corridor from the camera coast (Miura/Izu
// peninsulas) to Mt. Fuji. Every vertex is placed at its TRUE geographic
// position relative to the wave-patch center (lon0, lat0): +x east,
// -z north, meters. The mountains therefore stand on the actual shoreline
// instead of floating as a distant tile board.
// ============================================================================
#include "terrain.h"
#include <cstdio>
#include <cmath>
#include <vector>

bool loadFujiDEM(const char* rgbPath, const char* metaPath,
                 float lon0, float lat0, float heightExag, int decimate,
                 TerrainMesh& out)
{
    // --- meta: lonMin lonMax latMin latMax W H ---
    FILE* fm = fopen(metaPath, "r");
    if (!fm) { fprintf(stderr, "no %s\n", metaPath); return false; }
    float lonMin, lonMax, latMin, latMax;
    int W, H;
    if (fscanf(fm, "%f %f %f %f %d %d", &lonMin, &lonMax,
               &latMin, &latMax, &W, &H) != 6) {
        fclose(fm);
        return false;
    }
    fclose(fm);

    FILE* fr = fopen(rgbPath, "rb");
    if (!fr) { fprintf(stderr, "no %s\n", rgbPath); return false; }
    std::vector<unsigned char> rgb((size_t)W * H * 3);
    if (fread(rgb.data(), 1, rgb.size(), fr) != rgb.size()) {
        fclose(fr);
        return false;
    }
    fclose(fr);

    const float mPerDegLat = 111000.0f;
    const float mPerDegLon = 111000.0f * cosf(35.3f * 3.14159265f / 180.0f);

    const int gw = W / decimate, gh = H / decimate;
    out.gridW = gw; out.gridH = gh;
    out.verts.resize((size_t)gw * gh * 4);

    auto elevAt = [&](int x, int y) {
        const unsigned char* p = &rgb[((size_t)y * W + x) * 3];
        return (float)p[0] * 256.0f + (float)p[1] + (float)p[2] / 256.0f
               - 32768.0f;
    };

    for (int j = 0; j < gh; ++j)
        for (int i = 0; i < gw; ++i) {
            const int sx = i * decimate, sy = j * decimate;
            const float lon = lonMin + (lonMax - lonMin) * (float)sx / (float)W;
            const float lat = latMax - (latMax - latMin) * (float)sy / (float)H;
            float e = elevAt(sx, sy);
            // sink sub-sea-level cells so the far-sea plane always covers
            // water; the shoreline stays where the DEM says it is
            if (e < 2.0f) e = -3.0f;
            else          e *= heightExag;
            float* v = &out.verts[((size_t)j * gw + i) * 4];
            v[0] = (lon - lon0) * mPerDegLon;        // east offset (m)
            v[1] = e;
            v[2] = -(lat - lat0) * mPerDegLat;       // +z = south (m)
            v[3] = e;                                // shading elevation
        }

    for (int j = 0; j < gh - 1; ++j)
        for (int i = 0; i < gw - 1; ++i) {
            const unsigned a = j * gw + i, b = a + 1;
            const unsigned c = a + gw, d = c + 1;
            out.idx.insert(out.idx.end(), { a, c, b, b, c, d });
        }
    return true;
}
