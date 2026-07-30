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
    // physical diameter projected to pixels with an artistic telephoto exaggeration boost (18.0x)
    // and a robust minimum point size of 3.8 pixels so clusters remain visible as physical spray mist.
    gl_PointSize = clamp(uPointScale * max(aSize, 0.015) * 18.0 / dist, 3.8, uMaxPx);

    // screen-space sun azimuth: where the highlight sits on each hollow film
    vec4 sp = uViewProj * vec4(normalize(uSunDir), 0.0);
    vSunXY = normalize(sp.xy + vec2(1e-6, 0.0));

    // Volumetric scattering illumination for flying droplets and rafted bubbles.
    // Instead of using unstable, high-frequency discrete surface normals that cause 
    // violent flickering as particles move across grid cells, 
    // we use a stable elevation-based occlusion model. 
    // Droplets high in the air catch full sun; droplets low in wave troughs are shadowed.
    vec2 uv = aPosAlpha.xz / uDomain + 0.5;
    float baseElev = texture(uHeight, uv).r;
    float altitude = max(aPosAlpha.y - baseElev, 0.0);
    float exposure = smoothstep(0.0, 3.0, altitude + 0.5); 
    vLight = 0.35 + 0.65 * exposure;
}
