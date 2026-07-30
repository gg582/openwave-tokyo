#version 330 core
// ============================================================================
// Far sea plane: the open bay water between the animated wave patch and the
// distant shore, so the coastline never floats. Flat (sub-pixel swell at
// that distance), same water model as the patch: Jerlov body color, water
// Fresnel sky reflection, sun glitter, aerial haze.
// ============================================================================
in vec3 vWorld;
in float vElev;
out vec4 fragColor;

uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uSunColor;

void main()
{
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 Hv = normalize(V + L);

    float NoV = clamp(dot(N, V), 0.0, 1.0);
    float F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
    vec3 R = reflect(-V, N);
    vec3 skyRef = mix(vec3(0.130, 0.270, 0.500),
                      vec3(0.680, 0.700, 0.690),
                      pow(clamp(1.0 - R.y, 0.0, 1.0), 2.0));

    float NoH = clamp(dot(N, Hv), 0.0, 1.0);
    float glint = pow(NoH, 600.0) * 1.2;

    float NoL = max(dot(N, L), 0.0);
    vec3 deep = vec3(0.008, 0.055, 0.095);
    vec3 col = deep * (0.6 + 0.4 * NoL) * 1.5 + F * skyRef * 0.75
             + uSunColor * glint;

    float dist = length(uCamPos - vWorld);
    float haze = 1.0 - exp(-dist * 0.00002);
    col = mix(col, vec3(0.600, 0.630, 0.645), haze);

    fragColor = vec4(col, 1.0);
}
