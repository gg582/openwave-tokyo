// ============================================================================
// climate.h — era-appropriate sun and weather from open data
//
// Two sources:
//   1. NASA POWER climatology (assets/climate.json, fetched from
//      https://power.larc.nasa.gov) for the scene site (35.2 N, 139.7 E):
//      solar irradiance, cloud amount, temperature, wind speed by month.
//      Note: instrumental weather records do not reach back to the 1830s
//      (the print's era); climatological norms are the defensible open
//      data for "the weather Hokusai would have known".
//   2. Astronomical solar position (NOAA-style approximation, ~0.01 deg)
//      computed for the assumed scene moment: a spring morning in 1831,
//      07:30 JST — the low eastern sun of the print.
// ============================================================================
#pragma once

struct SceneClimate {
    float solarIrradiance = 4.11f;   // kWh/m^2/day (March, site)
    float cloudAmount     = 0.607f;  // 0..1 (March, site)
    float temperatureC    = 11.2f;   // deg C (March, site)
    float windSpeed       = 6.0f;    // m/s at 10 m (March, site)
    float fittedWildWind  = 25.0f;   // Fitted severe storm base wind speed (m/s)
    float gustMultiplier  = 1.85f;   // Fitting factor for peak typhoon gusts
    float sunDir[3]       = {0.9f, 0.2f, 0.38f};  // toward the sun (world)
    float sunColor[3]     = {1.0f, 0.90f, 0.72f};
    float sunIntensity    = 1.0f;
};

// Reads assets/climate.json (naive key search, no JSON library), computes
// the solar position for 1831-03-21 07:30 JST at (lat, lon), and fits severe wind parameters.
SceneClimate loadSceneClimate(const char* jsonPath, float lat, float lon, const char* month = "MAR");
