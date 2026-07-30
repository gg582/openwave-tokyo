#version 330 core
// Spray droplet point sprites: one GL point per droplet
layout(location = 0) in vec4 aPosAlpha;   // xyz = world pos, w = alpha

uniform mat4  uViewProj;
uniform float uPointScale;                // px per meter at 1 m distance

out float vAlpha;

void main()
{
    vAlpha = aPosAlpha.w;
    gl_Position = uViewProj * vec4(aPosAlpha.xyz, 1.0);
    float dist = max(gl_Position.w, 1.0);
    gl_PointSize = clamp(uPointScale * 0.07 / dist, 1.0, 22.0);  // ~7 cm drops
}
