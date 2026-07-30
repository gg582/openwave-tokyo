// ============================================================================
// tide.h — astronomical tide prediction for Tokyo Bay / Uraga Channel
//
// Harmonic method (the same scheme used by national tide tables):
//   H(t) = Z0 + sum_i  f_i * A_i * cos(V_i(t) + u_i - kappa_i)
// Constituent astronomical arguments V_i are evaluated from the mean
// longitudes of the moon (s), sun (h), lunar perigee (p) and ascending
// node (N), so the phases are correct for ANY epoch — including the
// 1831-03-21 07:30 JST scene date (lunar age 7.1 d, just before first
// quarter: a neap tide, small range, which the prediction reproduces).
// Amplitudes/phases approximate the published JMA constants for the
// Uraga / bay-mouth station (inner-bay amplification is weaker here).
// ============================================================================
#pragma once

// Julian Day from a proleptic-Gregorian calendar date + UT hour.
double tideJulianDay(int year, int month, int day, double hourUT);

// Predicted tide level above mean sea level (meters) at the given Julian
// Day (UT). Z0 = 0: heights are relative to MSL; Japanese chart datum
// sits ~0.95 m below MSL in Tokyo Bay if a CD reference is needed.
float tideLevelMSL(double jdUT);
