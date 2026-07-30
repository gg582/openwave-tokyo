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

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
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
    float NoL = clamp(dot(N, L), 0.0, 1.0);

    // 2. Triplanar Texturing
    vec3 blendWeights = pow(abs(N), vec3(4.0));
    blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z + 1e-4);

    float dist = length(uCamPos - vWorld);
    float scale = 0.0005;

    vec3 rockTex = sampleTriplanar(uRockTex, vWorld, blendWeights, scale);
    vec3 rockAlbedo = clamp(rockTex * 0.45, vec3(0.04), vec3(0.40));

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

    float fbm = noise(vWorld.xz * 0.001) * 0.5 + noise(vWorld.xz * 0.005) * 0.5;
    float elevGrad = smoothstep(2300.0, 300.0, vElev);
    float slopeGrad = smoothstep(0.40, 0.90, N.y);
    float vegDensity = elevGrad * slopeGrad * (0.4 + 0.6 * fbm);

    // Volcanic Radial Erosion Channels (Sharp gullies carved along volcanic slopes)
    // Use the normalized vertex normal direction (N.xz) to compute the radial polar angle.
    // This solves the floating-point precision loss and flickering bug caused by large offset subtractions on vWorld.xz!
    float polarAngle = atan(N.z, N.x);
    float erosionFbm = noise(vec2(polarAngle * 45.0, 0.0)) * 0.5
                     + noise(vec2(polarAngle * 95.0, 0.0)) * 0.25
                     + noise(vec2(polarAngle * 195.0, 0.0)) * 0.125;
    float slopeIntensity = smoothstep(0.35, 0.85, 1.0 - N.y);
    float erosionFactor = erosionFbm * slopeIntensity * smoothstep(1000.0, 3776.0, vElev);
    
    // Apply erosion weathering to soil/rock albedo (darkens the gullies)
    vec3 weatheredSoil = baseFujiSoil * (1.0 - erosionFactor * 0.45);
    vec3 terrainAlbedo = mix(weatheredSoil, fujiFloraAlbedo, clamp(vegDensity * (1.0 - erosionFactor * 0.3), 0.0, 0.85));

    // 4. Lighting
    float orenNayarDiffuse = NoL; 
    vec3 col = terrainAlbedo * (0.35 + 0.65 * orenNayarDiffuse);

    // 5. Physically-Guided Wind-Swept Snow Drifts
    // High-resolution multi-octave noise for jagged, windswept snow margins
    float snowFbm = noise(vWorld.xz * 0.015) * 0.5
                  + noise(vWorld.xz * 0.045) * 0.25
                  + noise(vWorld.xz * 0.125) * 0.125;
    
    // NW wind shears the snow coverage across the summit slopes (scale tuned to 35.0 to stay near peak)
    float windShear = dot(N.xz, vec2(-0.707, 0.707)) * 35.0;
    
    // Snow clings to upper valleys/erosion gullies, clears completely on steep rock cliffs (N.y)
    // Raising base snow line to 2550m ensures lower woodland, subalpine forests, and photographic rock textures are fully visible.
    float snowN = N.y * (1.3 + 0.2 * snowFbm) + erosionFactor * 0.3;
    float snowLine = 2550.0 + windShear + snowFbm * 110.0 - erosionFactor * 250.0;
    float snowAmt = smoothstep(snowLine - 80.0, snowLine + 80.0, vElev) * smoothstep(0.48, 0.78, snowN);
    snowAmt = clamp(snowAmt, 0.0, 1.0);
    
    // Micro wind-blown ripples on the snow surface
    float snowRipples = noise(vWorld.xz * 0.15) * 0.15 + noise(vWorld.xz * 0.45) * 0.05;
    vec3 snowC = vec3(0.94, 0.96, 0.98) + snowRipples;
    col = mix(col, snowC * (0.75 + 0.25 * NoL), snowAmt);

    // 6. Volumetric Exponential Height Fog (Low-altitude sea mist hugs the bay, peaks pierce the clouds)
    float fogDensity = 0.000028;
    float heightFalloff = 0.0016; // mist thins out exponentially as elevation climbs
    float fogHaze = 1.0 - exp(-dist * fogDensity * exp(-max(vWorld.y, 0.0) * heightFalloff));
    col = mix(col, uHorizonCol, fogHaze);

    fragColor = vec4(col, 1.0);
}
