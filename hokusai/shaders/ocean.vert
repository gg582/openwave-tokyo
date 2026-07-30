#version 330 core
// ============================================================================
// Ocean surface vertex stage: grid mesh displaced by the CUDA wave field.
//   height   : R32F  surface elevation (m), shoaling gain already applied
//   disp     : RG32F horizontal choppiness displacement (m)
// ============================================================================
layout(location = 0) in vec2 aGrid;     // [0,1]^2 grid coordinate

uniform sampler2D uHeight;
uniform sampler2D uDisp;
uniform mat4  uViewProj;
uniform float uDomain;                  // meters
uniform float uLambda;                  // extra choppiness scale
uniform float uTide;                    // instantaneous water level (m)

out vec3 vWorld;
out vec2 vUv;

void main()
{
    vec2 uv = aGrid;
    // Continuous bicubic/bilinear height sampling from CUDA texture
    float h = texture(uHeight, uv).r + uTide;
    vec2  d = texture(uDisp, uv).rg;

    // Continuous world position mapping without discrete grid offsets
    vec3 world = vec3((uv.x - 0.5) * uDomain,
                      h,
                      (uv.y - 0.5) * uDomain);
    world.xz += uLambda * d;

    vWorld = world;
    vUv = uv;
    gl_Position = uViewProj * vec4(world, 1.0);
}
