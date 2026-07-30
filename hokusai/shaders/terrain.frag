#version 330 core
// ============================================================================
// Real Mount Fuji terrain shading:
//   * photographed rock (CC0 Rock063) tiled in world space
//   * snow cover by elevation + slope with a ragged edge
//   * strong aerial perspective toward the horizon color (~70 km away)
// ============================================================================
in vec3 vWorld;
in float vElev;
out vec4 fragColor;

uniform sampler2D uRockTex;
uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uHorizonCol;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

void main()
{
    vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (N.y < 0.0) N = -N;
    vec3 L = normalize(uSunDir);
    float NoL = clamp(dot(N, L), 0.0, 1.0);

    // photographed rock in world-space tiling, natural warm-gray grade
    vec3 rockTex = texture(uRockTex, vWorld.xz * 0.0018).rgb;
    float lum = dot(rockTex, vec3(0.299, 0.587, 0.114));
    vec3 rock = rockTex * vec3(0.95, 0.90, 0.85) * (0.6 + 0.8 * lum);

    // snow cover: elevation-driven with ragged noise edge, thins on steeps
    float ragged = noise(vWorld.xz * 0.004) * 350.0;
    float snowLine = 2250.0 + ragged;
    float snow = smoothstep(snowLine - 120.0, snowLine + 120.0, vElev);
    snow *= smoothstep(0.25, 0.55, N.y);          // sheds on steep flanks
    // streaks running downhill
    snow = clamp(snow + 0.25 * smoothstep(0.6, 0.9,
                    noise(vWorld.xz * vec2(0.02, 0.002)))
                    * smoothstep(snowLine - 500.0, snowLine, vElev),
                 0.0, 1.0);
    vec3 snowC = vec3(0.905, 0.910, 0.900);

    vec3 col = mix(rock, snowC, snow) * (0.35 + 0.65 * NoL);

    // aerial perspective: ~70 km of atmosphere between camera and peak
    float dist = length(uCamPos - vWorld);
    float haze = 1.0 - exp(-dist * 0.000012);
    col = mix(col, uHorizonCol, haze);
    // lowlands melt into the hazy skyline so the DEM boundary never shows
    vec3 lowHaze = mix(uHorizonCol, vec3(0.30, 0.40, 0.45), 0.35);
    col = mix(col, lowHaze, smoothstep(45.0, 5.0, vElev));

    fragColor = vec4(col, 1.0);
}
