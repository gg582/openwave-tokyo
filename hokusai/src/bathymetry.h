// ============================================================================
// bathymetry.h — GEBCO-compatible bathymetry load / generate
//
// Depth convention follows GEBCO: `elevation` is meters relative to mean
// sea level, NEGATIVE below the surface. The struct exposes positive
// depth H(x,y) = -elevation for the physics engine.
//
// The NetCDF reader/writer implements the NetCDF-3 classic format directly
// (no netcdf library dependency). Real GEBCO grids (lat/lon/elevation,
// NC_FLOAT or NC_SHORT with scale_factor/add_offset) drop in unchanged;
// when no file is given, a procedural model of the Uraga Channel at the
// mouth of Tokyo Bay (35.2 N, 139.7 E) is synthesized instead.
// ============================================================================
#pragma once
#include <vector>

struct Bathymetry {
    int   width  = 0;       // samples along longitude (x, east)
    int   height = 0;       // samples along latitude  (y, north)
    float latMin = 0.f, latMax = 0.f;
    float lonMin = 0.f, lonMax = 0.f;
    std::vector<float> elevation;   // row-major [lat][lon], meters (neg = sea)
};

// Procedural Uraga Channel model (Tokyo Bay mouth, 35.2 N 139.7 E).
Bathymetry generateSyntheticUraga(int n);

// NetCDF-3 classic I/O.
bool loadNetCDF3(const char* path, Bathymetry& out);
bool writeNetCDF3(const char* path, const Bathymetry& b);
