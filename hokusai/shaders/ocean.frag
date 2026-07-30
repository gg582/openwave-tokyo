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

    // --- PHYSICAL OPTICS PATCH 1: Depth-dependent Beer-Lambert Wavelength Attenuation ---
    // Pure Jerlov Type II coastal water extinction coefficients (1/m: Red, Green, Blue)
    const vec3 JerlovII_Sigma = vec3(0.280, 0.045, 0.018);
    float bathyDepth = max(texture(uDepth, vUv).r, 0.5);
    float surfaceHgt = texture(uHeight, vUv).r;
    float totalWaterColumn = max(bathyDepth + max(surfaceHgt, 0.0), 0.2);

    // Path length accounting for slanted viewing angle (grazing ray penetration)
    float pathLen = totalWaterColumn / max(NoV, 0.12);
    vec3 waterTransmittance = exp(-JerlovII_Sigma * pathLen);

    // --- PHYSICAL OPTICS PATCH 2: Atmospheric Rayleigh Sky & Dynamic Ocean Illumination ---
    vec3 skyRadiance = uSunColor * vec3(0.14, 0.29, 0.54) * (0.80 + 0.35 * graze);

    // Deep water upwelling radiance (forward & backward ambient scattering in water column)
    vec3 deepUpwelling = skyRadiance * vec3(0.08, 0.22, 0.35) * (0.30 + 0.70 * NoL);

    // Shallow seabed reflectance attenuated by two-way water column path
    vec3 seabedAlbedo = vec3(0.28, 0.24, 0.18);
    if (uHasFoamTex == 1) {
        seabedAlbedo = texture(uFoamAlbedo, vWorld.xz * 0.08).rgb * vec3(0.60, 0.55, 0.45);
    }
    vec2 depthGrad = vec2(dFdx(bathyDepth), dFdy(bathyDepth));
    vec3 seabedIllum = uSunColor * NoL * seabedAlbedo * (1.0 - clamp(length(depthGrad) * 6.0, 0.0, 0.50));
    vec3 seabedReflectance = seabedIllum * exp(-JerlovII_Sigma * (2.0 * bathyDepth));

    // Continuous depth weighting without artificial step functions
    float seabedVisibility = exp(-0.18 * bathyDepth) * pow(1.0 - graze, 1.2);
    vec3 body = mix(deepUpwelling * waterTransmittance, seabedReflectance, seabedVisibility);

    // --- PHYSICAL OPTICS PATCH 3: Continuous Air Entrainment & Optical Foam Volume ---
    float airFraction = texture(uFoam, vUv).r;
    // Optical thickness of aerated foam (continuous physical exponential function)
    float aeratedOpacity = 1.0 - exp(-2.4 * max(airFraction, 0.0));

    // --- Cook-Torrance Specular Reflection --------------------------------
    float roughness = mix(0.04, 0.65, aeratedOpacity);
    vec3 F = F_Schlick(VoH, vec3(F0));
    float D = D_GGX(NoH, roughness);
    float G = G_Smith(float(NoV), NoL, roughness);
    vec3 specularReflectance = F * D * G / max(4.0 * NoV * NoL, 1e-4) * uSunColor * NoL;

    // Fresnel Sky Reflection
    vec3 R = reflect(-V, N);
    vec3 skyReflection = mix(skyRadiance * 0.6, skyRadiance * 1.4, pow(clamp(1.0 - R.y, 0.0, 1.0), 2.2));
    vec3 color = body + F * skyReflection * 1.05 + specularReflectance * 1.35;

    // --- PHYSICAL OPTICS PATCH 4: Henyey-Greenstein Volumetric Subsurface Scattering ---
    {
        float crestness = smoothstep(0.2, 5.0, max(surfaceHgt, 0.0));
        float cosTheta = dot(-V, L);
        // Henyey-Greenstein phase function (g = 0.72 forward scattering parameter for seawater)
        float g_hg = 0.72;
        float hgPhase = (1.0 - g_hg * g_hg) / pow(1.0 + g_hg * g_hg - 2.0 * g_hg * cosTheta, 1.5);

        vec3 sssTransmission = exp(-JerlovII_Sigma * 1.5) * uSunColor * hgPhase * 0.28;
        color += sssTransmission * crestness * (1.0 - aeratedOpacity * 0.75);
    }

    // --- PHYSICAL OPTICS PATCH 5: Continuous Sea Foam Light Scattering (No Sharp Gap) ---
    // Aerated bubbles scatter local incident irradiance without artificial stark color gaps
    vec3 foamNormal = N;
    if (uHasFoamTex == 1) {
        vec3 microNorm = texture(uFoamNormal, vUv * 7.0).rgb * 2.0 - 1.0;
        foamNormal = normalize(N + microNorm * 0.25);
    }
    float foamNoL = clamp(dot(foamNormal, L), 0.0, 1.0);
    vec3 incidentFoamIrradiance = uSunColor * (foamNoL * 0.85 + 0.15) + skyRadiance * 0.60;

    // Foam color is a continuous blend between aerated water body and light-scattering air volume
    vec3 aeratedWaterCol = body * 1.4 + sssTransmission * 0.5;
    vec3 whitecapScattering = incidentFoamIrradiance * 0.90;
    vec3 foamColor = mix(aeratedWaterCol, whitecapScattering, aeratedOpacity);

    // Continuous physical blend of foam onto water surface
    color = mix(color, foamColor, aeratedOpacity * 0.80);

    // --- PHYSICAL OPTICS PATCH 6: Entrained Micro-Bubbles (Continuous Glints) ---
    if (airFraction > 0.001) {
        vec2 buv = vWorld.xz * 6.0;
        vec2 cellId = floor(buv);
        vec2 f = fract(buv) - 0.5;
        float r1 = hash(cellId);
        float r2 = hash(cellId + vec2(37.0, 91.0));
        vec2 c = (vec2(r1, r2) - 0.5) * 0.7;
        float d2 = dot(f - c, f - c) * 4.84;
        float alive = 0.5 + 0.5 * sin(r1 * 13.7 + uTime * 2.5);
        float bubbleIntensity = exp(-d2 * 2.2) * alive;

        vec3 bubbleNormal = normalize(vec3((f - c) * 1.5, sqrt(max(1.0 - clamp(d2, 0.0, 0.99), 0.01))));
        float bubbleNoL = clamp(dot(bubbleNormal, L), 0.0, 1.0);
        vec3 bubbleSpecular = uSunColor * pow(clamp(dot(reflect(-L, bubbleNormal), V), 0.0, 1.0), 18.0);

        vec3 bubbleColor = mix(foamColor, incidentFoamIrradiance * bubbleNoL + bubbleSpecular, 0.35);
        color = mix(color, bubbleColor, bubbleIntensity * aeratedOpacity * 0.40);
    }

    // Atmospheric Distance Fog
    float cameraDist = length(uCamPos - vWorld);
    float atmosphericHaze = 1.0 - exp(-cameraDist * 0.00032);
    color = mix(color, skyRadiance * 1.15, atmosphericHaze);

    fragColor = vec4(color, 1.0);
}
