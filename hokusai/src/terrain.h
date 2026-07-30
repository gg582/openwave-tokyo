// ============================================================================
// terrain.h — real terrain from a Terrarium DEM, geographically anchored
// ============================================================================
#pragma once
#include <vector>

struct TerrainMesh {
    std::vector<float>    verts;    // pos.xyz + elevation, stride 4
    std::vector<unsigned> idx;
    int gridW = 0, gridH = 0;
};

// lon0/lat0: geographic origin of the world frame (wave-patch center).
bool loadFujiDEM(const char* rgbPath, const char* metaPath,
                 float lon0, float lat0, float heightExag, int decimate,
                 TerrainMesh& out);
