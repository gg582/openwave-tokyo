#version 330 core
// ============================================================================
// Ocean surface fragment stage — Sagami Bay / Tokyo Bay coastal water
//
//   * Beer-Lambert depth attenuation, green-window extinction
//     (chlorophyll/sediment-laden bay water)
//   * Cook-Torrance BRDF: GGX distribution, Smith geometry,
//     Schlick Fresnel with F0 = 0.02 (water)
//   * whitecap foam rendered as individual GPU bubble particles (spray);
//     the CUDA Jacobian/breaking mask only feeds the bubble emitter scan
//   * Prussian-blue Hokusai grade + distance haze
//
// References
// ----------
// [1] Cook R.L. & Torrance K.E. (1982). "A reflectance model for computer
//     graphics." ACM SIGGRAPH Computer Graphics, 15(3), 307–316.
//     → Cook-Torrance BRDF: D_GGX, G_Smith, F_Schlick (directSunSpecular)
//
// [2] Schlick C. (1994). "An inexpensive BRDF model for physically-based
//     rendering." Computer Graphics Forum, 13(3), 233–246.
//     → Fresnel approximation F_Schlick + exact polarized Fresnel (water n=1.333)
//
// [3] Henyey L.C. & Greenstein J.L. (1941). "Diffuse radiation in the galaxy."
//     Astrophysical Journal, 93, 70–83.
//     → Multi-lobe HG phase function for subsurface scattering (SSS section)
//
// [4] Jerlov N.G. (1976). Marine Optics. Elsevier, Amsterdam.
//     → Jerlov IB water type: extinction coefficients sigma_R/G/B (JerlovIB_Sigma)
//
// [5] Monahan E.C. & O'Muircheartaigh I. (1980). "Optimal power-law
//     description of oceanic whitecap coverage dependence on wind speed."
//     J. Physical Oceanography, 10(12), 2094–2099.
//     → Whitecap/shoreline foam spectral color and lifetime basis
//
// [6] Mitsuyasu H. et al. (1975). "Observations of the directional spectrum
//     of ocean waves using a cloverleaf buoy."
//     J. Physical Oceanography, 5(4), 750–760.
//     → Directional spreading informs near-camera normal blend (camDist2D)
// ============================================================================
in vec3 vWorld;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uHeight;
uniform sampler2D uDisp;
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
    // Screen-space faceted normal (micro-facet detail from mesh silhouette)
    vec3 Nface = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (Nface.y < 0.0) Nface = -Nface;

    // Height-field spectral normal: analytically derived from the CUDA wave texture.
    // This is the ground truth for the wave shape and dominates at close range
    // where screen-space derivatives become unreliable (camera nearly perpendicular
    // to the surface makes dFdx/dFdy approach zero, collapsing Nface to (0,1,0)).
    vec2 texel = 1.0 / vec2(textureSize(uHeight, 0));
    float hL = texture(uHeight, vUv - vec2(texel.x, 0.0)).r;
    float hR = texture(uHeight, vUv + vec2(texel.x, 0.0)).r;
    float hD = texture(uHeight, vUv - vec2(0.0, texel.y)).r;
    float hU = texture(uHeight, vUv + vec2(0.0, texel.y)).r;
    float cell = uDomain / float(textureSize(uHeight, 0).x);
    vec3 Ns = normalize(vec3(-(hR - hL) / (2.0 * cell), 1.0,
                             -(hU - hD) / (2.0 * cell)));

    // At long range, blend in a little Nface for sub-texel micro-facet glitter.
    // At close range (< 80 m) rely entirely on Ns to prevent the flat 2D-texture look.
    float camDist2D = length(uCamPos.xz - vWorld.xz);
    float blendFacet = smoothstep(20.0, 120.0, camDist2D) * 0.15;
    vec3 N = normalize(mix(Ns, Nface, blendFacet));

    // Close-range surface detail: mix analytic spectral normal with micro-ripple
    // normals from CC0 Foam001 material and high-frequency procedural wind ripples.
    if (uHasFoamTex == 1) {
        // Break up tiling with irrational rotations and disparate scales
        vec2 uvA = mat2( 0.866,  0.500, -0.500,  0.866) * vWorld.xz * 0.32 + vec2(uTime * 0.015);
        vec2 uvB = mat2( 0.707, -0.707,  0.707,  0.707) * (vWorld.xz + vWorld.y * 0.2) * 0.75 - vec2(uTime * 0.025);
        
        vec3 mnA = texture(uFoamNormal, uvA).rgb * 2.0 - 1.0;
        vec3 mnB = texture(uFoamNormal, uvB).rgb * 2.0 - 1.0;
        
        // Large-scale organic surface detail via noise gradient
        vec2 cp = vWorld.xz * 1.25 + uTime * 0.45;
        float n = noise(cp) + 0.4 * noise(cp * 2.5);
        vec2 grad = vec2(noise(cp + 0.1) - n, noise(cp + vec2(0, 0.1)) - n) * 5.0;
        
        // Procedural micro-ripple normal (creates the shimmering silver wind-glitter across the bay)
        // Replaced simple sine/cosine grids with an irrational rotation multi-frequency detail set
        // to avoid periodic moire artifacts (cloth-like pattern) while capturing high-frequency wind ripples.
        vec2 ripUv1 = mat2( 0.940,  0.342, -0.342,  0.940) * vWorld.xz * 2.6 - uPropDir * uTime * 2.1;
        vec2 ripUv2 = mat2( 0.574, -0.819,  0.819,  0.574) * vWorld.xz * 5.8 - uPropDir * uTime * 3.4;
        vec2 ripUv3 = mat2(-0.174,  0.985, -0.985, -0.174) * vWorld.xz * 11.2 - uPropDir * uTime * 5.1;
        
        float ripA = noise(ripUv1) * 0.45 + noise(ripUv2) * 0.25;
        float ripB = noise(ripUv3) * 0.12 + sin(ripUv1.x * 1.5 + ripUv2.y * 0.8) * 0.06;
        vec2 microGrad = vec2(ripA - ripB * 0.4, ripB + ripA * 0.2) * 0.35;
        
        float mnStrength = mix(0.40, 0.15, smoothstep(20.0, 150.0, camDist2D));
        float ripStrength = mix(0.24, 0.06, smoothstep(40.0, 600.0, camDist2D));
        
        N = normalize(N + vec3(mnA.x + 0.7 * mnB.x + grad.x, 0.0,
                               mnA.z + 0.7 * mnB.z + grad.y) * mnStrength
                        + vec3(microGrad.x, 0.0, microGrad.y) * ripStrength);
    }

    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 H = normalize(V + L);

    float NoV = clamp(dot(N, V), 1e-4, 1.0);
    // NoL: keep full range for diffuse/SSS, but guard against zero in specular denom.
    // Micro-normal perturbation can flip facets toward sun even when macro-NoL < 0;
    // clamping to 1e-4 (not 0) preserves those glittering specular hits.
    float NoL = clamp(dot(N, L), 1e-4, 1.0);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    // View-angle term used by the optical path-length and shallow-water
    // contribution below.  This must be defined before either is evaluated;
    // the missing declaration made this shader fail to link, leaving only the
    // flat far-sea pass visible in the encoded video.
    float graze = clamp(1.0 - NoV, 0.0, 1.0);

    // --- 1. PHYSICAL OCEAN WATER OPTICS (1831 Edo Bay / Sagami Estuary) ---
    // High marine microbial life (phytoplankton) and suspended river minerals.
    // Enhanced attenuation in blue and red (chlorophyll + CDOM), yielding a richer green-blue tone.
    // sigma_R ~ 0.280, sigma_G ~ 0.045, sigma_B ~ 0.115 m^-1
    const vec3 EdoBay_Sigma = vec3(0.280, 0.045, 0.115); 

    float bathyDepth = max(texture(uDepth, vUv).r, 0.05);
    float surfaceHgt = texture(uHeight, vUv).r;
    float totalWaterColumn = max(bathyDepth + max(surfaceHgt, 0.0), 0.01);

    // Two-way optical path length for volume transparency (Beer-Lambert attenuation)
    float pathLen = totalWaterColumn * (1.0 / max(NoV, 0.08) + 1.0 / max(NoL, 0.08));
    vec3 waterTransmittance = exp(-EdoBay_Sigma * pathLen);

    // Deep volume scattering coefficient (physical radiative transfer approximation)
    vec3 backscatterCoeff = vec3(0.015, 0.098, 0.145) * 0.35; 
    vec3 volumeScattering = backscatterCoeff * (1.0 - waterTransmittance) / (EdoBay_Sigma + 1e-4);
    vec3 deepUpwelling = uSunColor * volumeScattering * (0.50 + 0.50 * NoL);

    // Shallow Seabed & Shoreline Sand/Rock Illumination
    vec3 seabedAlbedo = vec3(0.16, 0.14, 0.11);
    if (uHasFoamTex == 1) {
        vec3 rockTex = texture(uFoamAlbedo, vWorld.xz * 0.06).rgb;
        vec3 mudTex  = texture(uFoamRough, vWorld.xz * 0.15).rgb;
        seabedAlbedo = mix(rockTex * vec3(0.25, 0.22, 0.18), mudTex * vec3(0.32, 0.28, 0.22), 0.5);
    }
    vec2 depthGrad = vec2(dFdx(bathyDepth), dFdy(bathyDepth));
    float wetSheen = exp(-max(totalWaterColumn - 0.2, 0.0) * 3.0);
    vec3 seabedIllum = uSunColor * (NoL * 0.85 + 0.15) * seabedAlbedo * (1.0 - clamp(length(depthGrad) * 3.0, 0.0, 0.35));
    seabedIllum += uSunColor * pow(NoH, 32.0) * 0.35 * wetSheen;

    // Physical Seabed Transparency
    float seabedVisibility = exp(-0.5 * totalWaterColumn) * (1.0 - graze);
    vec3 body = mix(deepUpwelling, seabedIllum, seabedVisibility);
    body = mix(body, body * waterTransmittance, 0.8);

    // --- 2. HIGH-INTENSITY SUN SPECULAR ---
    // Crest-dependent Roughness: Crests are rougher due to wind and micro-breaking, while troughs are glass-like.
    float crestFactor = smoothstep(0.4, 3.5, max(surfaceHgt, 0.0));
    float roughness = mix(0.04, 0.18, crestFactor) + 0.12 * graze;

    // Polarized Fresnel (Fresnel equations, n=1.338 for Edo Bay estuarine seawater, ~32 psu)
    float eta = 1.0 / 1.338;
    float sinThetaT2 = eta * eta * (1.0 - VoH * VoH);
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT2));
    float Rs = ((VoH - 1.338 * cosThetaT) / (VoH + 1.338 * cosThetaT));
    float Rp = ((1.338 * VoH - cosThetaT) / (1.338 * VoH + cosThetaT));
    float fresnelExact = clamp(0.5 * (Rs * Rs + Rp * Rp), 0.02, 0.98);
    vec3 F = mix(vec3(0.02), vec3(1.0), fresnelExact);

    // Anisotropic GGX Specular along wind propagation direction to simulate realistic water glint stretching
    vec3 T_aniso = normalize(vec3(uPropDir.x, 0.0, uPropDir.y)); 
    vec3 B_aniso = cross(N, T_aniso);
    float roughX = max(roughness * 0.45, 0.01); // Narrower in wind direction
    float roughY = max(roughness * 1.45, 0.01); // Wider crosswind
    float XoH = dot(T_aniso, H);
    float YoH = dot(B_aniso, H);
    float d_denom = XoH*XoH / (roughX*roughX) + YoH*YoH / (roughY*roughY) + NoH*NoH;
    float D_aniso = 1.0 / (3.14159265 * roughX * roughY * d_denom * d_denom + 1e-5);

    float G = G_Smith(float(NoV), NoL, roughness);
    float macroNoL = clamp(dot(normalize(cross(dFdx(vWorld), dFdy(vWorld))), L), 0.0, 1.0);
    vec3 directSunSpecular = F * D_aniso * G / max(4.0 * NoV * NoL, 1e-4)
                             * uSunColor * max(macroNoL, 0.05) * 10.0;

    // Dynamic sky radiance reflection
    vec3 skyRadiance = uSunColor * mix(vec3(0.25, 0.50, 0.85), vec3(0.42, 0.66, 0.95), graze);
    vec3 R = reflect(-V, N);
    vec3 skyReflection = mix(skyRadiance * 0.90, skyRadiance * 1.25, pow(clamp(1.0 - R.y, 0.0, 1.0), 2.0));
    vec2 gp = vWorld.xz + vec2(vWorld.y * 0.8, -vWorld.y * 0.6);
    float glitter = 0.70 + 0.85 * noise(gp * 2.5 + vec2(uTime * 0.12, -uTime * 0.08))
                         * noise(gp * 0.65 - vec2(uTime * 0.04));
    skyReflection *= glitter;
    vec3 color = body + F * skyReflection + directSunSpecular;

    // --- 2.5. PHYSICAL FLUID-ADVECTED FOAM BLENDING ---
    float foamAmt = texture(uFoam, vUv).r;
    
    // Branchless smooth foam blending to prevent hard snap flickering at foamAmt thresholds
    float activeFoam = smoothstep(0.005, 0.06, foamAmt);
    
    vec2 fluidDisp = texture(uDisp, vUv).rg;
    // Turbulence & wave slope-driven foam stretching along gravity/slope normal to simulate shearing
    vec2 slopeStretch = Ns.xz * (0.35 * smoothstep(0.0, 1.5, length(Ns.xz)));
    vec2 advectedUV = vWorld.xz + fluidDisp * 1.5 + slopeStretch * 4.5; 
    
    vec2 uvFoamA = mat2( 0.866,  0.500, -0.500,  0.866) * advectedUV * 0.05 + vec2(uTime * 0.005);
    vec2 uvFoamB = mat2( 0.707, -0.707,  0.707,  0.707) * advectedUV * 0.12 - vec2(uTime * 0.012);
    
    float foamOpA = texture(uFoamOpacity, uvFoamA).r;
    float foamOpB = texture(uFoamOpacity, uvFoamB).r;
    float textureOpacity = mix(foamOpA, foamOpB, 0.5);
    
    float foamOpacity = smoothstep(0.1, 0.8, foamAmt) * textureOpacity * 2.8 * activeFoam;
    foamOpacity = clamp(foamOpacity, 0.0, 1.0);
    
    vec3 foamAlbA = texture(uFoamAlbedo, uvFoamA).rgb;
    vec3 foamAlbB = texture(uFoamAlbedo, uvFoamB).rgb;
    vec3 foamAlbedo = mix(foamAlbA, foamAlbB, 0.5);
    
    // Volumetric thickness self-shadowing based on height and light direction (no offset sampling to prevent cloth/moire alignment)
    float foamSelfShadow = mix(0.40, 1.0, clamp(NoL * (1.0 + 0.6 * textureOpacity), 0.0, 1.0));
    vec3 foamScatter = (uSunColor * (0.45 + 0.55 * NoL) + vec3(0.12, 0.18, 0.25)) * foamSelfShadow; 
    vec3 foamDiffuse = foamAlbedo * foamScatter * 1.15; 
    
    // Blend foam continuously without hard if-conditionals
    color = mix(color, foamDiffuse, foamOpacity * 0.85 * float(uHasFoamTex));

    // --- 3. SHORELINE DEPTH FADE ---
    // Extended range (0.05 → 2.5 m) gives a gradual alpha fade that eliminates
    // the hard bright-band artifact at the terrain/water mesh seam.
    // The inner limit 0.05 m avoids a hard cutoff exactly at the mesh edge.
    float depthFade = smoothstep(0.05, 2.5, totalWaterColumn);
    // Shoreline foam: neutral warm-white rather than raw sunColor to prevent
    // the bright cyan/green fringe when sunColor has a strong tint.
    vec3 shoreFoamColor = mix(vec3(0.92, 0.90, 0.88),          // breaking-wave white
                              uSunColor * vec3(1.0, 0.95, 0.88), // warm sun tint
                              0.25);                             // 25 % tint, 75 % white
    float shoreFoam = exp(-totalWaterColumn * 2.2)
                      * (0.5 + 0.5 * noise(vWorld.xz * 0.5 + uTime * 0.2));
    color = mix(seabedIllum, color, depthFade) + shoreFoamColor * shoreFoam * 0.50;

    // --- MULTI-FORWARD HENYEY-GREENSTEIN VOLUMETRIC SUBSURFACE SCATTERING ---
    {
        float crestness = smoothstep(0.4, 3.5, max(surfaceHgt, 0.0));
        float cosTheta = dot(-V, L);

        // Forward lobe (g = 0.72) + Secondary lobe (g = 0.40) for organic wave-crest transillumination
        float g1 = 0.72, g2 = 0.40;
        float hg1 = (1.0 - g1 * g1) / pow(max(0.01, 1.0 + g1 * g1 - 2.0 * g1 * cosTheta), 1.5);
        float hg2 = (1.0 - g2 * g2) / pow(max(0.01, 1.0 + g2 * g2 - 2.0 * g2 * cosTheta), 1.5);
        float hgPhase = clamp(0.7 * hg1 + 0.3 * hg2, 0.0, 3.2);

        // Wave-front Light Leak: Enhanced transillumination through steep wave fronts facing away from the sun
        // Based on local wave crest thickness estimation
        float waveThickness = max(0.05, 3.5 - max(surfaceHgt, 0.0));
        float sssAlign = clamp(dot(-Ns, L), 0.0, 1.0);
        float waveFrontLeak = exp(-EdoBay_Sigma.g * waveThickness * 1.5) * sssAlign * 3.5;
        // Organic Hokusai-style deep blue-green SSS (Edo bay nutrient-rich)
        vec3 sssTransmission = uSunColor * vec3(0.02, 0.32, 0.22) * (hgPhase * 0.18 + waveFrontLeak * 0.35);
        color += sssTransmission * crestness * (0.65 + 0.35 * noise(vWorld.xz * 0.8 + vWorld.y * 0.6));
    }

    // Atmospheric Distance Fog — converges to the SAME horizon fog tone the
    // background sky uses below the horizon, so near sea, far sea and sky
    // read as one continuous body instead of stacked bands
    vec3 horizonFog = vec3(0.42, 0.46, 0.52) * uSunColor * 0.7875;
    float cameraDist = length(uCamPos - vWorld);
    float atmosphericHaze = 1.0 - exp(-cameraDist * 0.00028);
    color = mix(color, horizonFog, atmosphericHaze);

    fragColor = vec4(color, 1.0);
}
