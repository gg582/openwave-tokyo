#version 330 core
// ============================================================================
// Post pass 1 — vintage-lens spherical aberration, vignetting & bloom.
// ============================================================================
in vec2 vNdc;
out vec4 fragColor;

uniform sampler2D uScene;
uniform float uAmount;        // aberration strength
uniform vec2  uTexel;

void main()
{
    vec2 uv = vNdc * 0.5 + 0.5;
    vec2 c  = uv - 0.5;
    float r2 = dot(c, c) * 4.0;         // 0 center .. ~2 corners
    float r4 = r2 * r2;

    // lateral chromatic aberration: per-channel radial dispersion
    float k = uAmount * 0.0035;
    vec2 offR = c * (k * r2);
    vec2 offB = -c * (k * r2);

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
        bloomAcc += max(col - vec3(0.88), vec3(0.0));
    }
    vec3 blur = acc / 8.0;
    vec3 bloom = (bloomAcc / 8.0) * 0.45;

    vec3 sharp;
    sharp.r = texture(uScene, uv + offR).r;
    sharp.g = texture(uScene, uv).g;
    sharp.b = texture(uScene, uv + offB).b;

    float w = clamp(uAmount * r4, 0.0, 1.0);
    vec3 result = mix(sharp, blur, w) + bloom;

    // Lens vignetting (natural peripheral light falloff)
    float vignette = 1.0 - 0.28 * pow(length(c * 1.35), 2.2);
    result *= clamp(vignette, 0.0, 1.0);

    fragColor = vec4(result, 1.0);
}
