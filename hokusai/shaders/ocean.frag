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

    // --- water body color: path-length Beer-Lambert transparency ---------
    // Thin near-surface water (lips, crests, shallow shore) transmits
    // light; the transparency collapses as the underwater path grows —
    // deeper columns and grazing rays turn opaque.
    float depth = texture(uDepth, vUv).r;              // bathymetric depth
    float hgt = texture(uHeight, vUv).r;
    float graze = clamp(1.0 - NoV, 0.0, 1.0);          // 0 = looking straight down
    float pathLen = max(depth + max(hgt, 0.0), 0.1)
                  * (0.35 + 2.2 * graze * graze);
    vec3 transm = exp(-SIGMA * pathLen);
    vec3 deepCol    = vec3(0.008, 0.055, 0.095);       // open-bay deep water
    vec3 shallowCol = vec3(0.030, 0.155, 0.155);       // sunlit coastal teal
    vec3 body = mix(deepCol, shallowCol, transm.g) * transm;
    // sandy bottom shows through thin shallow columns viewed steeply,
    // shaded by the actual seafloor relief (depth gradient)
    vec3 bottomCol = vec3(0.32, 0.30, 0.235);
    if (uHasFoamTex == 1) {
        // photographed rock detail as the seafloor material, sandy tint
        bottomCol = texture(uFoamAlbedo, vWorld.xz * 0.08).rgb
                  * vec3(0.85, 0.80, 0.65);
    }
    vec2 dg = vec2(dFdx(depth), dFdy(depth));
    bottomCol *= (0.75 + 0.5 * NoL) * (1.0 - clamp(length(dg) * 8.0, 0.0, 0.45));
    float bottomVis = clamp(transm.g * pow(1.0 - graze, 1.5)
                            * smoothstep(6.0, 1.5, depth), 0.0, 1.0);
    body = mix(body, bottomCol, bottomVis * 0.8);

    // --- whitecap: air-entrainment model ----------------------------------
    // A = air fraction from the CUDA entrainment equation. Three layers,
    // following what breaking surf actually does (and the print shows):
    //   1) a solid bright core where air keeps accumulating (the lip)
    //   2) streaky trails draining down the face along propagation
    //   3) spray droplets scattering off the lip
    float A = texture(uFoam, vUv).r;
    float core = smoothstep(0.55, 0.85, A);
    vec2 suv = vec2(dot(vUv, uPropDir) * 1.1,
                    dot(vUv, vec2(-uPropDir.y, uPropDir.x)) * 5.0);
    float streak = 1.0;
    if (uHasFoamTex == 1)
        // the photographed opacity map is dark (mean ~0.2): use it for the
        // lace SHAPE only, remapped — never as a brightness multiplier
        streak = smoothstep(0.08, 0.45, texture(uFoamOpacity, suv).r);
    float mid = smoothstep(0.15, 0.55, A) * (1.0 - core) * streak;
    float spray = smoothstep(0.60, 0.95, noise(vUv * 260.0 + uTime * 0.05))
                * smoothstep(0.35, 0.75, A) * (1.0 - core);
    float foamVis = clamp(core + mid * 0.85 + spray * 0.5, 0.0, 1.0);

    // --- Cook-Torrance specular ------------------------------------------
    // roughness: glossy open water -> rough foam (CC0 roughness map)
    float rough = 0.04;
    if (uHasFoamTex == 1) {
        float fr = texture(uFoamRough, vUv * 7.0).r;
        rough = mix(0.04, clamp(fr, 0.35, 0.95), foamVis);
    }
    vec3 F = F_Schlick(VoH, vec3(F0));
    float D = D_GGX(NoH, rough);
    float G = G_Smith(float(NoV), NoL, rough);
    vec3 spec = F * D * G / max(4.0 * NoV * NoL, 1e-4) * uSunColor * NoL;

    // --- sky reflection through the same Fresnel term ---------------------
    vec3 R = reflect(-V, N);
    vec3 skyRef = mix(vec3(0.110, 0.240, 0.480),
                      vec3(0.720, 0.760, 0.790),
                      pow(clamp(1.0 - R.y, 0.0, 1.0), 2.2));
    vec3 color = body * (0.30 + 0.70 * NoL) * 1.3 + F * skyRef * 1.45
               + spec * 2.2
               + vec3(0.015, 0.022, 0.028);   // ambient skylight floor

    // --- Volumetric Subsurface Scattering (SSS) --------------------------
    // Real wave lips pass backlit morning sunlight through thin crests
    // with forward phase scattering (Henyey-Greenstein type phase).
    {
        float crestH = texture(uHeight, vUv).r;
        float crestness = smoothstep(0.5, 6.0, crestH);
        float forwardScatter = pow(clamp(dot(-V, L), 0.0, 1.0), 3.0);
        float sideTranslucency = pow(clamp(1.0 - NoV, 0.0, 1.0), 2.0);
        vec3 sssColor = vec3(0.04, 0.42, 0.36) * uSunColor;
        color += sssColor * (forwardScatter * 2.5 + sideTranslucency * 0.8) * crestness * (1.0 - foamVis * 0.7);
    }

    // --- whitecap foam (CUDA air-fraction field + CC0 material) -----------
    // environment-lit: bubbles multiply-scatter the ACTUAL scene light —
    // the 1831 sun plus sky ambient — with the optical depth of the
    // aerated layer setting how opaque the foam is. Never painted white.
    vec3 foamTex = vec3(0.95, 0.97, 0.97);
    if (uHasFoamTex == 1) {
        foamTex = texture(uFoamAlbedo, vUv * 7.0).rgb / 0.40;
        vec3 fn = texture(uFoamNormal, vUv * 7.0).rgb * 2.0 - 1.0;
        foamTex *= 0.8 + 0.2 * fn.z;
    }
    vec3 skyAmb = mix(vec3(0.130, 0.270, 0.500),
                      vec3(0.680, 0.700, 0.690), 0.5);
    float thick = 1.0 - exp(-1.8 * clamp(A, 0.0, 1.5));
    vec3 foamLight = uSunColor * (0.35 + 0.65 * NoL) * 1.1 + skyAmb * 0.45;
    vec3 foamCol = foamTex * foamLight * (0.35 + 0.75 * thick);
    color = mix(color, foamCol, foamVis);

    // --- individual air bubbles (the entrained air, rendered one by one) --
    // cellular field in world space: each cell carries ONE spherical bubble
    // with its own dome glint and rim, popping in and out over its lifetime
    if (foamVis > 0.01) {
        vec2 buv = vWorld.xz * 6.0;                 // ~17 cm bubbles
        vec2 cellId = floor(buv);
        vec2 f = fract(buv) - 0.5;
        float r1 = hash(cellId);
        float r2 = hash(cellId + vec2(37.0, 91.0));
        vec2 c = (vec2(r1, r2) - 0.5) * 0.7;
        float d = length(f - c) * 2.2;              // 0 = center, 1 = rim
        float alive = step(fract(r1 * 13.7 + uTime * 0.4), 0.75);
        float sphere = smoothstep(1.0, 0.75, d) * alive;
        float dome = pow(clamp(1.0 - d, 0.0, 1.0), 2.5);
        vec3 bubbleCol = foamCol * (0.85 + 0.15 * r2) + vec3(0.35) * dome;
        color = mix(color, bubbleCol, sphere * foamVis * 0.85);
    }

    // --- distance haze into the backdrop ---------------------------------
    float dist = length(uCamPos - vWorld);
    float haze = 1.0 - exp(-dist * 0.00035);
    color = mix(color, vec3(0.600, 0.630, 0.645), haze);

    fragColor = vec4(color, 1.0);
}
