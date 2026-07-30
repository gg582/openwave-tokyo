#version 330 core
// One spray droplet: spherical water bead lit by the environment
// (1831 sun + sky ambient), never painted white
in float vAlpha;
out vec4 fragColor;

uniform vec3 uSunColor;

void main()
{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0 || vAlpha <= 0.001) discard;
    float dome = sqrt(1.0 - r2);                 // spherical profile
    vec3 skyAmb = vec3(0.30, 0.36, 0.42);
    vec3 col = uSunColor * (0.30 + 0.70 * dome) * 1.05 + skyAmb * 0.35;
    float a = smoothstep(1.0, 0.82, r2) * vAlpha;
    fragColor = vec4(col, a);
}
