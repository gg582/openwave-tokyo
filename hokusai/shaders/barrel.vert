#version 330 core
// Barrel ribbon vertices: world-space positions generated per frame.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aInfo;   // x = tau (jet param), y = rib

uniform mat4 uViewProj;

out vec3 vWorld;
out float vTau;
out float vRib;

void main()
{
    vWorld = aPos;
    vTau = aInfo.x;
    vRib = aInfo.y;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
