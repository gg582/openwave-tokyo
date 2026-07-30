#version 330 core
// ============================================================================
// Barrel lip ribbon — physically derived plunging-jet surface.
//
// Cross-sections follow the ballistic trajectory of the ejected lip
// (Longuet-Higgins & Cokelet plunging-jet model): the crest launches a jet
// at roughly the wave phase speed and the sheet curls over under gravity.
// The ribbon is generated per frame from the actual crest line of the
// spectral wave field — no sculpted static shapes.
// ============================================================================
in vec3 vWorld;
in float vTau;        // 0 = jet origin (crest), 1 = jet tip (landing)
in float vRib;        // position along the crest line (for variation)
out vec4 fragColor;

uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform float uTime;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main()
{
    vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    if (dot(N, V) < 0.0) N = -N;         // thin sheet: face the camera

    float NoL = clamp(dot(N, L), 0.0, 1.0);
    float NoV = clamp(dot(N, V), 1e-4, 1.0);
    vec3 H = normalize(V + L);
    float NoH = clamp(dot(N, H), 0.0, 1.0);

    // water body of the falling sheet: translucent green, thinner at tip
    vec3 body = vec3(0.020, 0.130, 0.150) * (0.45 + 0.55 * NoL);
    float through = pow(clamp(dot(-V, L), 0.0, 1.0), 2.0);
    body += vec3(0.04, 0.32, 0.28) * through * (1.0 - 0.5 * vTau);

    // fresnel sky reflection
    float F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
    vec3 R = reflect(-V, N);
    vec3 skyRef = mix(vec3(0.130, 0.270, 0.500),
                      vec3(0.680, 0.700, 0.690),
                      pow(clamp(1.0 - R.y, 0.0, 1.0), 3.0));

    // specular glint from the 1831 sun
    float spec = pow(NoH, 120.0) * 0.8;

    // foam: only the outer lip and the aerated jet tip — the sheet itself
    // stays translucent water
    float foamAmt = smoothstep(0.35, 0.10, vTau) * 0.9
                  + smoothstep(0.80, 0.95, vTau) * 0.8;
    // environment-lit foam (scattering of the actual sun + sky, not paint)
    vec3 foamLight = uSunColor * (0.45 + 0.55 * NoL)
                   + vec3(0.35, 0.42, 0.50) * 0.5;
    float granule = 0.85 + 0.3 * hash(floor(vWorld.xz * 14.0));
    vec3 foamCol = foamLight * granule * 0.62;

    vec3 col = body + F * skyRef * 0.9 + uSunColor * spec;
    col = mix(col, foamCol, clamp(foamAmt, 0.0, 1.0));

    // aerial haze
    float dist = length(uCamPos - vWorld);
    col = mix(col, vec3(0.600, 0.630, 0.645), 1.0 - exp(-dist * 0.00035));

    fragColor = vec4(col, 1.0);
}
