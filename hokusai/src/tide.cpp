// ============================================================================
// tide.cpp — harmonic tide prediction, see tide.h
// ============================================================================
#include "tide.h"
#include <cmath>

double tideJulianDay(int year, int month, int day, double hourUT)
{
    int y = year, m = month;
    if (m <= 2) { y -= 1; m += 12; }
    const int A = y / 100;
    const int B = 2 - A + A / 4;
    return (double)(int)(365.25 * (y + 4716))
         + (double)(int)(30.6001 * (m + 1))
         + day + hourUT / 24.0 + B - 1524.5;
}

namespace {

struct Constituent {
    double speed;   // deg per mean solar hour
    double A;       // amplitude (m), Uraga / Tokyo Bay mouth
    double kappa;   // phase lag (deg)
};

// M2, S2, N2, K2 (semidiurnal), K1, O1, P1 (diurnal).
// Amplitudes/lags representative of the Uraga fairway: slightly weaker and
// earlier than inner-bay stations (Tokyo/Yokohama), co-tidal charts show
// M2 ~0.35 m propagating in from Sagami-nada.
constexpr Constituent C[] = {
    { 28.9841042, 0.350, 115.0 },   // M2  principal lunar
    { 30.0000000, 0.120, 145.0 },   // S2  principal solar
    { 28.4397295, 0.070, 100.0 },   // N2  lunar elliptic
    { 30.0821373, 0.035, 145.0 },   // K2  lunisolar
    { 15.0410686, 0.200, 160.0 },   // K1  lunisolar diurnal
    { 13.9430356, 0.150, 140.0 },   // O1  lunar diurnal
    { 14.9589314, 0.070, 160.0 },   // P1  solar diurnal
};
constexpr int NC = sizeof(C) / sizeof(C[0]);

constexpr double DEG = 3.14159265358979323846 / 180.0;

} // namespace

float tideLevelMSL(double jdUT)
{
    // Julian centuries from J2000.0 and hours since the start of the day
    const double T  = (jdUT - 2451545.0) / 36525.0;
    const double h24 = (jdUT - std::floor(jdUT - 0.5) - 0.5) * 24.0; // UT hour

    // Mean longitudes (deg), Meeus low-precision set — the same arguments
    // Schureman-style harmonic prediction is built on
    const double s = std::fmod(218.3164477 + 481267.88123421 * T, 360.0);
    const double h = std::fmod(280.4660757 + 36000.76982779 * T, 360.0);
    const double p = std::fmod( 83.3532430 +  4069.01371110 * T, 360.0);
    const double N = std::fmod(125.0445550 -  1934.13618490 * T, 360.0);

    // Astronomical argument V0 at t=0h for each constituent
    const double V0[NC] = {
        std::fmod(2 * h - 2 * s + 360.0, 360.0),        // M2
        0.0,                                            // S2
        std::fmod(2 * h - 3 * s + p + 360.0, 360.0),    // N2
        std::fmod(2 * h, 360.0),                        // K2
        std::fmod(h + 90.0, 360.0),                     // K1
        std::fmod(h - 2 * s - 90.0 + 720.0, 360.0),     // O1
        std::fmod(-h + 270.0 + 360.0, 360.0),           // P1
    };

    // Nodal corrections (f, u) — only the dominant lunar terms matter at
    // this accuracy level
    const double Nd = N * DEG;
    const double fM2 = 1.0004 - 0.0373 * std::cos(Nd)
                     + 0.0002 * std::cos(2 * Nd);
    const double uM2 = (-2.14 * std::sin(Nd)) * DEG;
    const double fK1 = 1.0060 + 0.1150 * std::cos(Nd)
                     - 0.0088 * std::cos(2 * Nd);
    const double uK1 = (-8.86 * std::sin(Nd) + 0.68 * std::sin(2 * Nd)) * DEG;
    const double fO1 = 1.0089 + 0.1871 * std::cos(Nd)
                     - 0.0147 * std::cos(2 * Nd);
    const double uO1 = (10.80 * std::sin(Nd) - 1.34 * std::sin(2 * Nd)) * DEG;

    double H = 0.0;
    for (int i = 0; i < NC; ++i) {
        double f = 1.0, u = 0.0;
        if (i == 0)      { f = fM2; u = uM2; }
        else if (i == 4) { f = fK1; u = uK1; }
        else if (i == 5) { f = fO1; u = uO1; }
        const double phase = (V0[i] + C[i].speed * h24 - C[i].kappa) * DEG + u;
        H += f * C[i].A * std::cos(phase);
    }
    return (float)H;
}
