#version 330 core
// Real Mount Fuji terrain (Terrarium DEM mesh)
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aElev;

uniform mat4  uViewProj;
uniform float uTide;        // 0 for land, tide level for the far sea plane

out vec3 vWorld;
out float vElev;

void main()
{
    vWorld = aPos + vec3(0.0, uTide, 0.0);
    vElev = aElev;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
