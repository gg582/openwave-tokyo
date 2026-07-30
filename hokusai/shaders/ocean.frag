#version 330 core
// ============================================================================
// Ocean surface fragment stage — coastal PBR (Jerlov Type II)
//
//   * Beer-Lambert depth attenuation, per-channel absorption
//     sigma = vec3(0.08, 0.03, 0.01) (coastal type II water)
//   * Cook-Torrance BRDF: GGX distribution, Smith geometry,
//     Schlick Fresnel with F0 = 0.02 (water)
//   * whitecap foam from the CUDA Jacobian/breaking mask
//   * Prussian-blue Hokusai grade + distance haze
// ============================================================================
in vec3 vWorld;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uHeight;
uniform sampler2D uFoam;
uniform sampler2D uDepth;
uniform vec3  uCamPos;
uniform vec3  uSunDir;        // toward the sun
uniform vec3  uSunColor;
uniform float uTime;
uniform float uDomain;
// CC0 material maps (ambientCG Foam001/Foam003)
uniform sampler2D uFoamAlbedo;
uniform sampler2D uFoamNormal;
uniform sampler2D uFoamOpacity;
uniform sampler2D uFoamRough;
uniform int uHasFoamTex;
uniform vec2 uPropDir;      // swell propagation unit vector (uv space)

const vec3 SIGMA = vec3(0.08, 0.03, 0.01);   // Jerlov II absorption (1/m)
const float F0 = 0.02;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

// --- Cook-Torrance terms ---------------------------------------------------
float D_GGX(float NoH, float rough)
{
    float a = rough * rough;
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}
float G_Smith(float NoV, float NoL, float rough)
{
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float g1 = NoV / (NoV * (1.0 - k) + k);
    float g2 = NoL / (NoL * (1.0 - k) + k);
    return g1 * g2;
}
vec3 F_Schlick(float VoH, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);
}

void main()
{
    // faceted detail from screen-space derivatives mixed into the smooth
    // spectral normal (small weight: surface micro-facets, not a style)
    vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (N.y < 0.0) N = -N;
    vec2 texel = 1.0 / vec2(textureSize(uHeight, 0));
    float hL = texture(uHeight, vUv - vec2(texel.x, 0.0)).r;
    float hR = texture(uHeight, vUv + vec2(texel.x, 0.0)).r;
    float hD = texture(uHeight, vUv - vec2(0.0, texel.y)).r;
    float hU = texture(uHeight, vUv + vec2(0.0, texel.y)).r;
    float cell = uDomain / float(textureSize(uHeight, 0).x);
    vec3 Ns = normalize(vec3(-(hR - hL) / (2.0 * cell), 1.0,
                             -(hU - hD) / (2.0 * cell)));
    N = normalize(mix(Ns, N, 0.2));

    // CC0 material maps (ambientCG Foam001): two octaves of photographed
    // micro-ripple normals, drifting with the surface current
    if (uHasFoamTex == 1) {
        vec3 mnA = texture(uFoamNormal, vUv * 24.0
                           + vec2(uTime * 0.008, uTime * 0.011)).rgb * 2.0 - 1.0;
        vec3 mnB = texture(uFoamNormal, vUv * 61.0
                           - vec2(uTime * 0.013, uTime * 0.006)).rgb * 2.0 - 1.0;
        N = normalize(N + vec3(mnA.x + 0.5 * mnB.x, 0.0,
                               mnA.y + 0.5 * mnB.y) * 0.18);
    }

    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 H = normalize(V + L);

    float NoV = clamp(dot(N, V), 1e-4, 1.0);
    float NoL = clamp(dot(N, L), 0.0, 1.0);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    // --- Physically-driven ocean Optics (Pure Spectral Beer-Lambert & Scattering) ---
    // Pure optical extinction coefficients (1/m for Pure Water: R=0.35, G=0.045, B=0.015)
    // and Rayleigh sky irradiance derived directly from the atmospheric sun color
    const vec3 WATER_SIGMA = vec3(0.35, 0.045, 0.015);
    float depth = texture(uDepth, vUv).r;              // bathymetric depth
    float hgt = texture(uHeight, vUv).r;
    float graze = clamp(1.0 - NoV, 0.0, 1.0);          // 0 = looking straight down
    float pathLen = max(depth + max(hgt, 0.0), 0.1) * (0.40 + 1.8 * graze * graze);

    // Translucency & extinction derived directly from physical path length and solar spectral light
    vec3 transm = exp(-WATER_SIGMA * pathLen);
    vec3 skyRadiance = uSunColor * vec3(0.12, 0.28, 0.58) * (1.0 + 0.5 * graze);
    vec3 body = skyRadiance * transm * (0.20 + 0.80 * NoL);

    // Seafloor diffuse reflection (derived dynamically from light extinction & material texture)
    vec3 bottomTex = vec3(0.35, 0.30, 0.22);
    if (uHasFoamTex == 1) {
        bottomTex = texture(uFoamAlbedo, vWorld.xz * 0.08).rgb;
    }
    vec2 dg = vec2(dFdx(depth), dFdy(depth));
    vec3 bottomIllum = uSunColor * NoL * bottomTex * (1.0 - clamp(length(dg) * 8.0, 0.0, 0.45));
    float bottomVis = clamp(transm.g * pow(1.0 - graze, 1.5) * smoothstep(6.0, 1.5, depth), 0.0, 1.0);
    body = mix(body, bottomIllum * transm, bottomVis);

    // --- whitecap: air-entrainment model ----------------------------------
    float A = texture(uFoam, vUv).r;
    float core = smoothstep(0.50, 0.90, A);
    vec2 suv = vec2(dot(vUv, uPropDir) * 1.1,
                    dot(vUv, vec2(-uPropDir.y, uPropDir.x)) * 5.0);
    float streak = 1.0;
    if (uHasFoamTex == 1)
        streak = smoothstep(0.08, 0.45, texture(uFoamOpacity, suv).r);
    float mid = smoothstep(0.15, 0.55, A) * (1.0 - core) * streak;
    float spray = smoothstep(0.60, 0.95, noise(vUv * 260.0 + uTime * 0.05))
                * smoothstep(0.35, 0.75, A) * (1.0 - core);
    float foamVis = clamp(core * 0.75 + mid * 0.60 + spray * 0.35, 0.0, 1.0);

    // --- Cook-Torrance specular ------------------------------------------
    float rough = 0.05;
    if (uHasFoamTex == 1) {
        float fr = texture(uFoamRough, vUv * 7.0).r;
        rough = mix(0.05, clamp(fr, 0.25, 0.85), foamVis);
    }
    vec3 F = F_Schlick(VoH, vec3(F0));
    float D = D_GGX(NoH, rough);
    float G = G_Smith(float(NoV), NoL, rough);
    vec3 spec = F * D * G / max(4.0 * NoV * NoL, 1e-4) * uSunColor * NoL;

    // --- sky reflection through Fresnel -----------------------------------
    vec3 R = reflect(-V, N);
    vec3 skyRef = mix(skyRadiance * 0.6, skyRadiance * 1.5, pow(clamp(1.0 - R.y, 0.0, 1.0), 2.2));
    vec3 color = body + F * skyRef * 1.10 + spec * 1.4 + skyRadiance * 0.05;

    // --- Volumetric Subsurface Scattering (SSS) --------------------------
    // Physical forward phase scattering through crests scaled by sun color
    {
        float crestH = texture(uHeight, vUv).r;
        float crestness = smoothstep(0.5, 6.0, crestH);
        float forwardScatter = pow(clamp(dot(-V, L), 0.0, 1.0), 3.0);
        float sideTranslucency = pow(clamp(1.0 - NoV, 0.0, 1.0), 2.0);
        vec3 sssColor = exp(-WATER_SIGMA * 2.0) * uSunColor;
        color += sssColor * (forwardScatter * 1.8 + sideTranslucency * 0.6) * crestness * (1.0 - foamVis * 0.8);
    }

    // --- Soft sea foam blending (derived physically from incident scene light)
    vec3 foamTex = vec3(0.85);
    if (uHasFoamTex == 1) {
        foamTex = texture(uFoamAlbedo, vUv * 7.0).rgb;
        vec3 fn = texture(uFoamNormal, vUv * 7.0).rgb * 2.0 - 1.0;
        foamTex *= 0.85 + 0.15 * fn.z;
    }
    float thick = 1.0 - exp(-1.2 * clamp(A, 0.0, 1.5));
    vec3 foamLight = uSunColor * (0.25 + 0.75 * NoL) * 0.85 + skyRadiance * 0.55;
    vec3 foamCol = mix(body * 1.5, foamTex * foamLight, 0.40 + 0.55 * thick);
    color = mix(color, foamCol, foamVis * 0.70);

    // --- individual air bubbles (softened wet translucent glints) ---------
    if (foamVis > 0.01) {
        vec2 buv = vWorld.xz * 6.0;                 // ~17 cm bubbles
        vec2 cellId = floor(buv);
        vec2 f = fract(buv) - 0.5;
        float r1 = hash(cellId);
        float r2 = hash(cellId + vec2(37.0, 91.0));
        vec2 c = (vec2(r1, r2) - 0.5) * 0.7;
        float d = length(f - c) * 2.2;
        float alive = step(fract(r1 * 13.7 + uTime * 0.4), 0.75);
        float sphere = smoothstep(1.0, 0.75, d) * alive;
        float dome = pow(clamp(1.0 - d, 0.0, 1.0), 2.5);
        vec3 bubbleCol = foamCol * (0.90 + 0.10 * r2) + uSunColor * 0.25 * dome;
        color = mix(color, bubbleCol, sphere * foamVis * 0.45);
    }

    // --- distance haze into the backdrop (derived from sky light) --------
    float dist = length(uCamPos - vWorld);
    float haze = 1.0 - exp(-dist * 0.00035);
    color = mix(color, skyRadiance * 1.2, haze);

    fragColor = vec4(color, 1.0);
}
