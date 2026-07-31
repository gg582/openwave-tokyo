#version 330 core
// ============================================================================
// Post pass 1 — vintage-lens spherical aberration, vignetting & bloom.
// ============================================================================
in vec2 vNdc;
out vec4 fragColor;

uniform sampler2D uScene;
uniform float uAmount;        // aberration strength
uniform vec2  uTexel;         // size of one pixel in texture UV units (1/fboW, 1/fboH)
uniform vec2  uMaxUv;         // (viewportW / fboW, viewportH / fboH)

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
    vec2 uvViewport = vNdc * 0.5 + 0.5;
    vec2 uv = uvViewport * uMaxUv;
    vec2 c  = uvViewport - 0.5;
    float r2 = dot(c, c) * 4.0;         // 0 center .. ~2 corners
    float r4 = r2 * r2;

    // lateral chromatic aberration: per-channel radial dispersion (disabled)
    float k = 0.0;
    vec2 offR = vec2(0.0);
    vec2 offB = vec2(0.0);

    // spherical blur: 8-tap Poisson disc, radius ~ r^4 (sharp center)
    float rad = uAmount * r4 * 1.5;
    const vec2 taps[8] = vec2[8](
        vec2( 0.707,  0.707), vec2(-0.707,  0.707),
        vec2( 0.707, -0.707), vec2(-0.707, -0.707),
        vec2( 1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0));
    vec3 acc = vec3(0.0);
    vec3 bloomAcc = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        vec2 t = taps[i] * rad * uTexel * 9.0;
        vec3 col;
        col.r = texture(uScene, uv + offR + t).r;
        col.g = texture(uScene, uv + t).g;
        col.b = texture(uScene, uv + offB + t).b;
        acc += col;
        bloomAcc += max(col - vec3(0.98), vec3(0.0));
    }
    vec3 blur = acc / 8.0;
    vec3 bloom = (bloomAcc / 8.0) * 0.15;

    vec3 sharp;
    sharp.r = texture(uScene, uv + offR).r;
    sharp.g = texture(uScene, uv).g;
    sharp.b = texture(uScene, uv + offB).b;

    // --- 1. SUBTLE CAMERA MOTION BLUR (Directional radial velocity blur) ---
    float w = clamp(uAmount * r4, 0.0, 1.0);
    vec2 velocityDir = normalize(c + vec2(1e-4)) * (0.0018 * uAmount);
    vec3 motionBlurAcc = vec3(0.0);
    const float mbTaps[4] = float[4](-1.5, -0.5, 0.5, 1.5);
    for (int i = 0; i < 4; ++i) {
        motionBlurAcc += texture(uScene, uv + velocityDir * mbTaps[i]).rgb;
    }
    vec3 result = mix(mix(sharp, blur, w), motionBlurAcc * 0.25, 0.35) + bloom;

    // --- 2. 1830s EDO PERIOD LITTLE ICE AGE ATMOSPHERIC GRAIN & SEA-VAPOR MIST ---
    // High-humidity Tokyo Bay sea-vapor aerosol mist tint (humid marine aerosol + pre-industrial clean air)
    vec3 edoSeaVaporMist = vec3(0.40, 0.44, 0.49);
    float seaVaporDensity = 0.18 + 0.08 * noise(uv * 12.0);
    result = mix(result, edoSeaVaporMist, seaVaporDensity);

    // Pre-industrial organic micro-dust grain noise (1830s organic pollen/ash dust, non-digital grain)
    float microDustGrain = (hash(uv * 400.0) - 0.5) * 0.012;
    result += vec3(microDustGrain);

    // FXAA (Fast Approximate Anti-Aliasing) Pass to remove sub-pixel temporal aliasing/jaggies
    vec2 rdx = vec2(uTexel.x, 0.0);
    vec2 rdy = vec2(0.0, uTexel.y);
    vec3 rgbNW = texture(uScene, uv - rdx - rdy).rgb;
    vec3 rgbNE = texture(uScene, uv + rdx - rdy).rgb;
    vec3 rgbSW = texture(uScene, uv - rdx + rdy).rgb;
    vec3 rgbSE = texture(uScene, uv + rdx + rdy).rgb;
    vec3 rgbM  = result;

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * 0.25), 0.0078125);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(8.0, 8.0), max(vec2(-8.0, -8.0), dir * rcpDirMin)) * uTexel;

    vec3 rgbA = 0.5 * (texture(uScene, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                      texture(uScene, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(uScene, uv + dir * -0.5).rgb +
                                    texture(uScene, uv + dir * 0.5).rgb);
    float lumaB = dot(rgbB, luma);

    if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
        result = rgbA;
    } else {
        result = rgbB;
    }

    // Lens vignetting (natural peripheral light falloff)
    float vignette = 1.0 - 0.28 * pow(length(c * 1.35), 2.2);
    result *= clamp(vignette, 0.0, 1.0);

    // Linear -> display gamma transform
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));

    fragColor = vec4(result, 1.0);
}
