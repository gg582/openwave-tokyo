#version 330 core
// Volumetric sea-foam splat & shredded spindrift chunk.
//
// Replaces static white colors with a physical spectral scattering model:
//   (A) Mie phase function scattering (forward peak)
//   (B) Jerlov Coastal II (Edo Bay) seawater spectral absorption 
//       (transmitting Prussian green-blue tint at wet boundaries)
in float vAlpha;
in float vSize;
in float vLight;
in vec2  vSunXY;
in float vEccentricity;
out vec4 fragColor;

uniform vec3 uSunColor;

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// 3-Octave Fractal Brownian Motion for fluid deformation
float fbm(vec2 p)
{
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; ++i) {
        v += a * noise(p);
        p *= 2.3;
        a *= 0.45;
    }
    return v;
}

void main()
{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    
    // --- 1. PHYSICAL OBLATE SQUEEZE (Weber-number & Taylor Analogy) ---
    vec2 flightDir = normalize(vec2(0.325, -0.946));
    vec2 perpDir = vec2(-flightDir.y, flightDir.x);
    float u_comp = dot(c, flightDir);
    float v_comp = dot(c, perpDir);
    
    float flatFactor = max(vEccentricity, 0.30);
    u_comp /= flatFactor;
    v_comp *= sqrt(flatFactor); // Mass/volume conservation
    vec2 deformedC = vec2(u_comp, v_comp);
    float dist = length(deformedC);
    
    // --- 2. KELVIN-HELMHOLTZ INSTABILITY (Velocity-aligned Shear Tearing) ---
    vec2 shearUV = vec2(u_comp * 0.40, v_comp * 2.80) + vec2(vSize * 8.5);
    float shearIntensity = 0.65 * (1.0 - vEccentricity);
    float edgeWarp = fbm(shearUV) * shearIntensity;
    
    float radialEdge = dist + edgeWarp - 0.20;
    if (radialEdge > 0.65 || vAlpha <= 0.001) discard;

    // --- 3. FLUID LIGAMENTS (Torn Aerodynamic Filaments) ---
    float weberFactor = 1.0 / max(vEccentricity, 0.25);
    float freqScale = 3.50 * sqrt(weberFactor);
    vec2 ligamentUV = vec2(u_comp * 0.60, v_comp * freqScale) - vec2(vSize * 15.0);
    float ligamentNoise = fbm(ligamentUV);
    float ligamentMask = smoothstep(0.28, 0.55, ligamentNoise);

    // --- 4. PHASE DISSOLUTION (Fine Spindrift Mist) ---
    float isMist = smoothstep(0.002, 0.0004, vSize);
    float opacityMap = mix(ligamentMask, 1.0 - radialEdge, isMist);

    // --- 5. PHYSICAL MULTIPLE-SCATTERING SPECTRAL COLORATION ---
    // (A) Jerlov Coastal II seawater spectral absorption (sigma_R=0.280, sigma_G=0.045, sigma_B=0.115 m^-1)
    vec3 Jerlov_Sigma = vec3(0.280, 0.045, 0.115);
    
    // Subsurface Scattering (SSS) ocean body color (Prussian Blue/Green)
    vec3 sssOceanColor = uSunColor * vec3(0.005, 0.16, 0.24) * vLight;
    
    // Mie scattering background sky ambient spectrum
    vec3 skyAmb = vec3(0.30, 0.36, 0.42) * uSunColor * 0.65;
    
    // Mie Phase Function approximation: forward scattering peaks on sunward side (g = 0.82)
    float cosTheta = dot(normalize(vec3(c, 1.0)), normalize(vec3(vSunXY, 0.5)));
    float phaseMie = 0.5 * (1.0 + cosTheta * cosTheta) / pow(1.0 + 0.82 * 0.82 - 2.0 * 0.82 * cosTheta, 1.5);
    vec3 directMieScatter = uSunColor * phaseMie * vLight;
    
    // Beer-Lambert wet boundary transmittance:
    // Thin wispy edges of the foam or mist absorb red/blue light, transmitting deep Prussian Blue/Green SSS,
    // while dense dry cores scatter sunlight fully to look bright warm-white.
    float pathLength = (1.0 - opacityMap) * 0.65; // thickness inverse
    vec3 wetFoamTransmittance = exp(-Jerlov_Sigma * pathLength * 12.0);
    
    // Combined spectral color: blend ocean body SSS color with dry Mie scattering
    vec3 dryFoamColor = mix(skyAmb, directMieScatter, 0.75);
    vec3 col = mix(sssOceanColor, dryFoamColor, wetFoamTransmittance);

    // Fade out smoothly towards the aerodynamically torn boundary
    float alphaFade = smoothstep(0.65, 0.15, radialEdge);
    float a = alphaFade * (0.35 + 0.65 * opacityMap) * vAlpha;

    if (a < 0.02) discard;
    fragColor = vec4(col, a);
}
