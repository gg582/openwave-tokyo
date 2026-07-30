#version 330 core
// Individual spray droplets & foam bubbles: one GL point per water bead,
// each with its own physical diameter (m) from the CUDA particle system.
// The bead's brightness follows the LOCAL wave slope lighting (sampled
// from the same height field the ocean mesh uses), so bubbles sitting on
// a shadowed breaking face dim with the water instead of glowing.
layout(location = 0) in vec4 aPosAlpha;   // xyz = world pos, w = alpha
layout(location = 1) in float aSize;      // droplet/bubble diameter (m)

uniform mat4  uViewProj;
uniform float uPointScale;                // px per meter at 1 m distance
uniform float uMaxPx;                     // resolution-proportional size cap
uniform sampler2D uHeight;
uniform float uDomain;
uniform vec3  uSunDir;

out float vAlpha;
out float vSize;
out float vLight;
out vec2  vSunXY;

void main()
{
    vAlpha = aPosAlpha.w;
    vSize = aSize;
    gl_Position = uViewProj * vec4(aPosAlpha.xyz, 1.0);
    float dist = max(gl_Position.w, 1.0);
    // physical diameter projected to pixels; tiny distant bubbles still get
    // a sub-pixel footprint so clusters read as texture, not a sheet
    gl_PointSize = clamp(uPointScale * max(aSize, 0.015) / dist, 1.0, uMaxPx);

    // screen-space sun azimuth: where the highlight sits on each hollow film
    vec4 sp = uViewProj * vec4(normalize(uSunDir), 0.0);
    vSunXY = normalize(sp.xy + vec2(1e-6, 0.0));

    // slope lighting from the height field (identical reconstruction to
    // ocean.frag's spectral normal)
    vec2 uv = aPosAlpha.xz / uDomain + 0.5;
    vec2 texel = 1.0 / vec2(textureSize(uHeight, 0));
    float hL = texture(uHeight, uv - vec2(texel.x, 0.0)).r;
    float hR = texture(uHeight, uv + vec2(texel.x, 0.0)).r;
    float hD = texture(uHeight, uv - vec2(0.0, texel.y)).r;
    float hU = texture(uHeight, uv + vec2(0.0, texel.y)).r;
    float cell = uDomain / float(textureSize(uHeight, 0).x);
    vec3 N = normalize(vec3(-(hR - hL) / (2.0 * cell), 1.0,
                            -(hU - hD) / (2.0 * cell)));
    float NoL = clamp(dot(N, normalize(uSunDir)), 0.0, 1.0);
    vLight = 0.30 + 0.70 * NoL;
}
