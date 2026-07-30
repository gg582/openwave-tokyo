// ============================================================================
// climate.cpp — NASA POWER climatology + astronomical solar position
// ============================================================================
#include "climate.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

// naive "KEY":number extraction (the POWER response is flat per parameter)
static float jsonFloat(const std::string& js, const char* key, float fallback)
{
    const std::string k = std::string("\"") + key + "\"";
    size_t p = js.find(k);
    if (p == std::string::npos) return fallback;
    p = js.find(':', p);
    if (p == std::string::npos) return fallback;
    return (float)atof(js.c_str() + p + 1);
}

// extract the MAR (March) value inside a parameter block
static float jsonMonth(const std::string& js, const char* param,
                       const char* month, float fallback)
{
    const std::string k = std::string("\"") + param + "\"";
    size_t p = js.find(k);
    if (p == std::string::npos) return fallback;
    size_t e = js.find('}', p);
    return jsonFloat(js.substr(p, e == std::string::npos ? e : e - p),
                     month, fallback);
}

// ---------------------------------------------------------------------------
// Solar ephemeris for the print's era, computed from first principles:
// Earth's orbital elements (mean longitude/anomaly, eccentricity,
// obliquity) and rotation (GMST) evaluated at the 1831 epoch via the
// standard low-precision series (Meeus, Astronomical Algorithms ch. 25 —
// valid to ~0.01 deg within a few centuries of J2000). This yields the
// apparent solar longitude, declination, right ascension, hour angle and
// the Sun-Earth distance for 1831-03-21 07:30 JST — rotation, revolution
// and solar distance all at their 1831 values.
// ---------------------------------------------------------------------------
static void solarPosition1831(float latDeg, float lonDeg,
                              float* elevDeg, float* azDeg, float* distAU)
{
    const double PI = 3.14159265358979;
    const double D2R = PI / 180.0, R2D = 180.0 / PI;

    // Julian date of 1831-03-20 22:30 UTC (= 1831-03-21 07:30 JST).
    // Double precision is mandatory: the series multiply the centuries
    // by ~36000 deg, and the result matters mod 360 to ~0.01 deg.
    const int Y = 1831, M = 3;
    const double D = 20.0 + 22.5 / 24.0;
    const int y = Y + 4800, m = M - 3;
    const double JD = D + (double)((153 * m + 2) / 5) + 365.0 * y
                    + (double)(y / 4) - (double)(y / 100) + (double)(y / 400)
                    - 32045.0 - 0.5;

    const double T = (JD - 2451545.0) / 36525.0;    // Julian centuries, J2000

    // orbital elements at 1831
    const double L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T;
    const double Mm = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
    const double e  = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;
    const double Mr = Mm * D2R;
    const double C  = (1.914602 - 0.004817 * T - 0.000014 * T * T) * sin(Mr)
                    + (0.019993 - 0.000101 * T) * sin(2 * Mr)
                    + 0.000289 * sin(3 * Mr);              // equation of center
    const double trueLong = L0 + C;
    const double trueAnom = Mm + C;
    const double omega = (125.04 - 1934.136 * T) * D2R;
    const double lambda = (trueLong - 0.00569 - 0.00478 * sin(omega)) * D2R;

    // Sun-Earth distance for 1831 (AU) and its irradiance factor
    const double R = 1.000001018 * (1.0 - e * e)
                   / (1.0 + e * cos(trueAnom * D2R));
    if (distAU) *distAU = (float)R;

    // obliquity of the ecliptic at 1831 (arcsecond series + nutation)
    const double epsArc = 84381.448 - 46.8150 * T - 0.00059 * T * T
                        + 0.001813 * T * T * T;
    const double eps = (epsArc / 3600.0 + 0.00256 * cos(omega)) * D2R;

    // equatorial coordinates
    const double RA  = atan2(cos(eps) * sin(lambda), cos(lambda));
    const double decl = asin(sin(eps) * sin(lambda));

    // Earth rotation: GMST at the 1831 epoch, then local hour angle
    const double d = JD - 2451545.0;
    double gmst = 280.46061837 + 360.98564736629 * d
                + 0.000387933 * T * T - T * T * T / 38710000.0;
    gmst = fmod(gmst, 360.0);
    if (gmst < 0) gmst += 360.0;
    const double HA = (gmst + lonDeg) * D2R - RA;

    // topocentric elevation/azimuth
    const double lat = latDeg * D2R;
    const double sinE = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(HA);
    const double elev = asin(sinE < -1 ? -1 : sinE > 1 ? 1 : sinE);
    const double cosAz = (sin(decl) - sin(lat) * sinE) / (cos(lat) * cos(elev));
    double az = acos(cosAz < -1 ? -1 : cosAz > 1 ? 1 : cosAz);
    if (sin(HA) > 0) az = 2.0 * PI - az;

    *elevDeg = (float)(elev * R2D);
    *azDeg   = (float)(az * R2D);
}

SceneClimate loadSceneClimate(const char* jsonPath, float lat, float lon, const char* month)
{
    SceneClimate c;
    std::ifstream f(jsonPath);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string js = ss.str();

    if (!js.empty()) {
        c.solarIrradiance = jsonMonth(js, "ALLSKY_SFC_SW_DWN", month,
                                      c.solarIrradiance);
        c.cloudAmount     = jsonMonth(js, "CLOUD_AMT", month,
                                      c.cloudAmount * 100.0f) / 100.0f;
        c.temperatureC    = jsonMonth(js, "T2M", month, c.temperatureC);
        c.windSpeed       = jsonMonth(js, "WS10M", month, c.windSpeed);
    }

    // Climate-driven wild wind fitting logic:
    // Scale baseline climatology wind speed to severe storm/typhoon wave generation regime.
    // WS50M / WS10M ratio ~ 1.25 (power law exponent ~0.14). Extreme 99th percentile gust factor ~ 3.2 - 4.5.
    c.fittedWildWind = fmaxf(c.windSpeed * 3.8f + (c.cloudAmount * 5.0f), 22.0f);
    c.gustMultiplier = 1.4f + 0.8f * c.cloudAmount + (c.windSpeed / 10.0f);

    float elev, az, distAU;
    solarPosition1831(lat, lon, &elev, &az, &distAU);

    // world: +x east, +y up, -z north; azimuth clockwise from north
    const float er = elev * 3.14159265f / 180.0f;
    const float ar = az * 3.14159265f / 180.0f;
    c.sunDir[0] = sinf(ar) * cosf(er);
    c.sunDir[1] = sinf(er);
    c.sunDir[2] = -cosf(ar) * cosf(er);

    // Sunlight color through the atmosphere at the computed elevation:
    // Beer-Bouguer extinction I = I0·exp(-tau·m), with the air mass m
    // from the Kasten-Young relation and per-channel extinction
    // (Rayleigh ~ 1/lambda^4 at 610/550/470 nm plus aerosol).
    const float elDeg = elev;
    const float m = 1.0f / (sinf(er) + 0.50572f
                  * powf(elDeg + 6.07995f, -1.6364f));

    // Dynamic Atmospheric Turbidity & Rayleigh Optical Depth linked to Japan Climatology
    // Lat 35.2N Marine Coastal Zone aerosol optical depth (AOD @ 550nm):
    const float marineAOD = 0.025f + 0.045f * c.cloudAmount + 0.005f * (c.temperatureC / 10.0f);
    const float tauA = marineAOD;

    float sR = expf(-(0.054f + tauA) * m);     // ~610 nm (Red Rayleigh)
    float sG = expf(-(0.086f + tauA) * m);     // ~550 nm (Green Rayleigh)
    float sB = expf(-(0.160f + tauA) * m);     // ~470 nm (Blue Rayleigh)

    // normalize so lighting stays usable while keeping the physical hue
    const float lum = 0.2126f * sR + 0.7152f * sG + 0.0722f * sB;
    const float nrm = 0.95f / fmaxf(lum, 1e-3f);
    c.sunColor[0] = sR * nrm;
    c.sunColor[1] = sG * nrm;
    c.sunColor[2] = sB * nrm;
    // irradiance: climatological value scaled by 1/R^2 and extinction
    c.sunIntensity = (c.solarIrradiance / 4.11f) / (distAU * distAU) * lum;

    printf("[climate] NASA POWER %s norms @ %.2fN %.2fE: solar %.2f kWh/m2/d, "
           "cloud %.0f%%, T %.1f C, wind %.1f m/s\n",
           month, lat, lon, c.solarIrradiance, c.cloudAmount * 100.0f,
           c.temperatureC, c.windSpeed);
    printf("[climate] Geographic Optical Air Mass: %.2f, AOD(550nm): %.3f -> Rayleigh Sun Tint (%.3f, %.3f, %.3f)\n",
           m, marineAOD, c.sunColor[0], c.sunColor[1], c.sunColor[2]);
    printf("[climate] Wild sea wind fitting: base %.1f m/s, peak gust mult %.2fx\n",
           c.fittedWildWind, c.gustMultiplier);
    printf("[climate] 1831-03-21 07:30 JST ephemeris: elevation %.1f deg, "
           "azimuth %.1f deg, Sun-Earth distance %.5f AU\n", elev, az, distAU);
    return c;
}
