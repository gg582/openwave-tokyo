#version 330 core
// ============================================================================
// Real Mount Fuji terrain shading:
//   * photographed rock (CC0 Rock063) tiled in world space
//   * snow cover by elevation + slope with a ragged edge
//   * strong aerial perspective toward the horizon color (~70 km away)
// ============================================================================
in vec3 vWorld;
in float vElev;
in vec3 vNormal;
out vec4 fragColor;

uniform sampler2D uRockTex;
uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform vec3  uHorizonCol;
uniform float uTime;

float hash(vec2 ip)
{
    uvec2 q = uvec2(ivec2(ip)) * uvec2(1597334977U, 3812015887U);
    uint n = (q.x ^ q.y) * 1597334977U;
    return float(n) * (1.0 / 4294967296.0);
}

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    // Quintic Hermite interpolation polynomial (C2 continuity) to eliminate slope banding snaps
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(hash(i),               hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

vec3 sampleTriplanar(sampler2D tex, vec3 p, vec3 weights, float sc)
{
    vec3 xT = texture(tex, p.yz * sc).rgb;
    vec3 yT = texture(tex, p.xz * sc).rgb;
    vec3 zT = texture(tex, p.xy * sc).rgb;
    return xT * weights.x + yT * weights.y + zT * weights.z;
}

void main()
{
    // 1. Smooth Analytic Normal
    vec3 N = normalize(vNormal);
    if (N.y < 0.0) N = -N;

    vec3 L = normalize(uSunDir);
    vec3 V = normalize(uCamPos - vWorld);
    // Half-Lambertian shading completely eliminates the snapping terminator edge, 
    // ensuring shadows never pop in/out abruptly on backfaces.
    float NoL_raw = dot(N, L);
    float NoL = NoL_raw * 0.5 + 0.5;
    NoL = NoL * NoL; // Squared for realistic contrast transition

    float dist = length(vec3(0.0, 100.0, -350.0) - vWorld);
    // 2. Triplanar Texturing (Faded to static flat projection in the distance to prevent normal jitter clicks)
    vec3 blendWeights = pow(abs(N), vec3(4.0));
    blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z + 1e-4);
    
    // Smoothly transition to flat Y-mapping beyond 15km where normal interpolation accuracy degrades
    float triplanarFade = clamp(1.0 - (dist / 15000.0), 0.0, 1.0);
    vec3 weights = mix(vec3(0.0, 1.0, 0.0), blendWeights, triplanarFade);
 
    float scale = 0.0005;
    vec3 rockTex = sampleTriplanar(uRockTex, vWorld, weights, scale);
    vec3 rockAlbedo = clamp(rockTex * 0.45, vec3(0.04), vec3(0.40));

    // Dynamic Micro-bump mapping: derive rock normals from texture luminance variation to add high-frequency volcanic scoria relief.
    float eps = 1.25; // spatial displacement step
    vec3 rockTexDX = sampleTriplanar(uRockTex, vWorld + vec3(eps, 0.0, 0.0), weights, scale);
    vec3 rockTexDZ = sampleTriplanar(uRockTex, vWorld + vec3(0.0, 0.0, eps), weights, scale);
    float luma = dot(rockTex, vec3(0.299, 0.587, 0.114));
    float lumaDX = dot(rockTexDX, vec3(0.299, 0.587, 0.114));
    float lumaDZ = dot(rockTexDZ, vec3(0.299, 0.587, 0.114));
    vec3 bumpNormal = normalize(vec3(luma - lumaDX, 0.15, luma - lumaDZ));
    // Completely fade out micro-bump normal mapping in the distance (triplanarFade reaches 0 at 15km)
    // to prevent sub-pixel normal jitter and triangle flickering on Mt. Fuji (70km away).
    N = normalize(N + bumpNormal * 0.24 * triplanarFade);

    // 3. Fuji Volcanic Soil
    vec3 kurobokudoSoil = rockAlbedo * vec3(0.35, 0.32, 0.28);
    vec3 akatsuchiScoria = rockAlbedo * vec3(0.65, 0.45, 0.35);
    float soilMix = noise(vWorld.xz * 0.005) * 0.5 + noise(vWorld.xz * 0.02) * 0.5;
    vec3 baseFujiSoil = mix(kurobokudoSoil, akatsuchiScoria, smoothstep(0.25, 0.75, soilMix));

    // Flora
    vec3 cedarAlbedo = rockAlbedo * vec3(0.25, 0.45, 0.25);
    vec3 oakAlbedo   = rockAlbedo * vec3(0.35, 0.55, 0.30);
    float speciesMix1 = noise(vWorld.xz * 0.015);
    vec3 lowWoodlandAlbedo = mix(cedarAlbedo, oakAlbedo, speciesMix1);

    vec3 veitchFirAlbedo = rockAlbedo * vec3(0.15, 0.35, 0.32);
    vec3 ermanBirchAlbedo = rockAlbedo * vec3(0.45, 0.60, 0.28);
    float speciesMix2 = noise(vWorld.xz * 0.025);
    vec3 subalpineConiferAlbedo = mix(veitchFirAlbedo, ermanBirchAlbedo, speciesMix2);

    float altFactor1 = smoothstep(300.0, 1500.0, vElev);
    float altFactor2 = smoothstep(1500.0, 2200.0, vElev);
    vec3 fujiFloraAlbedo = mix(lowWoodlandAlbedo, subalpineConiferAlbedo, altFactor1);

    vec3 fujiFloraAlbedo2 = mix(subalpineConiferAlbedo, baseFujiSoil * 0.8, altFactor2);
    fujiFloraAlbedo = mix(fujiFloraAlbedo, fujiFloraAlbedo2, altFactor2);

    float fbm = noise(vWorld.xz * 0.001) * 0.5 + noise(vWorld.xz * 0.005) * 0.5;
    float elevGrad = smoothstep(2300.0, 300.0, vElev);
    float slopeGrad = smoothstep(0.40, 0.90, N.y);
    float vegDensity = elevGrad * slopeGrad * (0.4 + 0.6 * fbm);

    // Volcanic Radial Erosion Channels (Sharp gullies carved along volcanic slopes)
    // Dynamic LOD: fade high-frequency noise bands at 70km distance to prevent sub-pixel popping.
    float lodFactor = clamp(1.0 - (dist / 85000.0), 0.0, 1.0);
    // Anchored polar angle to static world-space coordinates instead of dynamic normals
    // to completely eliminate high-frequency erosion noise snapping on sharp ridges.
    float polarAngle = atan(vWorld.z, vWorld.x);
    float erosionFbm = noise(vec2(polarAngle * 10.0, 0.0)) * 0.5
                     + noise(vec2(polarAngle * 24.0, 0.0)) * 0.25 * lodFactor
                     + noise(vec2(polarAngle * 54.0, 0.0)) * 0.125 * lodFactor * lodFactor;
    // Widened slope transition to prevent sharp ridge normal jitter from snapping color layers
    float slopeIntensity = smoothstep(0.10, 0.95, 1.0 - N.y);
    float erosionFactor = erosionFbm * slopeIntensity * smoothstep(1000.0, 3776.0, vElev);
    
    // Apply erosion weathering to soil/rock albedo (darkens the gullies)
    vec3 weatheredSoil = baseFujiSoil * (1.0 - erosionFactor * 0.45);
    vec3 terrainAlbedo = mix(weatheredSoil, fujiFloraAlbedo, clamp(vegDensity * (1.0 - erosionFactor * 0.3), 0.0, 0.85));

    // 4. Lighting
    float orenNayarDiffuse = NoL; 
    vec3 col = terrainAlbedo * (0.35 + 0.65 * orenNayarDiffuse);

    // 5. Physically-Guided Wind-Swept Snow Drifts
    // High-resolution multi-octave noise adapted with distance LOD to guarantee zero temporal flickering
    float snowFbm = noise(vWorld.xz * 0.02) * 0.5
                  + noise(vWorld.xz * 0.08) * 0.3 * lodFactor
                  + noise(vWorld.xz * 0.24) * 0.2 * lodFactor * lodFactor;
    
    // NW wind shears the snow coverage across the summit slopes.
    // Use the smooth unperturbed normal (smoothN) to completely avoid geometry-snapping flickers.
    vec3 smoothN = normalize(vNormal);
    if (smoothN.y < 0.0) smoothN = -smoothN;
    float windShear = dot(smoothN.xz, vec2(-0.707, 0.707)) * 80.0;
    
    // Snow clings deeply into upper valleys/erosion gullies (erosionFactor) and clears on steep cliffs (smoothN.y)
    float snowN = smoothN.y * (1.1 + 0.3 * snowFbm) + erosionFactor * 0.45;
    
    // Raise the base snow line from 3150m to 3150m to confine snow to the top peak zone.
    float snowLine = 3150.0 + windShear + snowFbm * 280.0 - erosionFactor * 350.0;
    // Widened transition width from 120.0 to 250.0 to guarantee buttery-smooth gradients without flickers
    float snowAmt = smoothstep(snowLine - 250.0, snowLine + 250.0, vElev) * smoothstep(0.40, 0.72, snowN);
    snowAmt = clamp(snowAmt, 0.0, 1.0);
    
    // Micro wind-blown ripples on the snow surface - smoothed with LOD
    float snowRipples = noise(vWorld.xz * 0.06) * 0.15 + noise(vWorld.xz * 0.18) * 0.05 * lodFactor;
    // Subsurface Scattering (SSS) inside the snow layer - subtle blue/cyan light transmission (using smoothN to stop popping)
    vec3 snowSSS = vec3(0.08, 0.18, 0.28) * clamp(dot(-smoothN, L), 0.0, 1.0) * (0.35 + 0.65 * snowFbm);
    vec3 snowC = vec3(0.92, 0.94, 0.96) + snowRipples;
    vec3 snowLighting = snowC * (0.35 + 0.65 * NoL) + snowSSS * uSunColor * 0.40;
    col = mix(col, snowLighting, snowAmt);

    // 6. Wet Shoreline (Wet sand & dark damp rock effect near sea level)
    // Water level oscillates slightly with uTime to simulate wave washing
    float waveOsc = sin(uTime * 0.8) * 0.45 + cos(uTime * 1.5) * 0.20;
    float shorelineFactor = smoothstep(12.0 + waveOsc, -2.0 + waveOsc, vElev);
    
    // Wet surface becomes darker and less saturated (absorbing water)
    col = mix(col, col * vec3(0.38, 0.35, 0.32), shorelineFactor);
    
    // Add specular sheen/reflection to wet shoreline
    float wetSheenShore = shorelineFactor * pow(max(dot(N, normalize(L + V)), 0.0), 32.0) * 0.28;
    col += uSunColor * wetSheenShore;

    // Volumetric Exponential Height Fog (Smoothed max transition at sea level to eliminate C0 derivative flickers)
    float fogDensity = 0.000028;
    float heightFalloff = 0.0016; 
    float smoothY = 0.5 * (vWorld.y + sqrt(vWorld.y * vWorld.y + 400.0)); // C1 continuous soft-max
    // Soften silhouette edges against the sky by blending long-range fog towards 1.0
    float fogHaze = 1.0 - exp(-dist * fogDensity * exp(-smoothY * heightFalloff));
    fogHaze = mix(fogHaze, 1.0, smoothstep(45000.0, 90000.0, dist) * 0.15); // Smoothly fade distant silhouette borders
    col = mix(col, uHorizonCol, fogHaze);

    fragColor = vec4(col, 1.0);
}
