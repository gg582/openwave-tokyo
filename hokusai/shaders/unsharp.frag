#version 330 core
// ============================================================================
// Post pass 2 (TRADITIONAL control) — separable Gaussian unsharp masking.
// Two invocations: horizontal blur, then vertical blur + combine
//   out = clamp(orig + amount * (orig - blur))
// Classic texture-fetch implementation: every tap goes through the memory
// hierarchy; no register-level exchange anywhere. This is the control group
// for the CUDA warp-shuffle inverse-OTF kernel.
// ============================================================================
in vec2 vNdc;
out vec4 fragColor;

uniform sampler2D uImage;     // input for the blur
uniform sampler2D uOrig;      // original (combine pass only)
uniform vec2  uDir;           // (texel.x, 0) or (0, texel.y)
uniform float uAmount;        // unsharp strength on the combine pass
uniform int   uCombine;       // 0 = blur only, 1 = blur + unsharp combine

void main()
{
    vec2 uv = vNdc * 0.5 + 0.5;
    const float w[5] = float[5](0.06136, 0.24477, 0.38774, 0.24477, 0.06136);
    vec3 acc = vec3(0.0);
    for (int i = -2; i <= 2; ++i)
        acc += w[i + 2] * texture(uImage, uv + uDir * float(i)).rgb;

    if (uCombine == 1) {
        vec3 orig = texture(uOrig, uv).rgb;
        acc = clamp(orig + uAmount * (orig - acc), 0.0, 1.0);
    }
    fragColor = vec4(acc, 1.0);
}
