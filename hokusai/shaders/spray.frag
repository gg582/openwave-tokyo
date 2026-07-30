#version 330 core
// One discrete water bead (spray droplet or foam bubble).
//
// A real sea-foam bubble is HOLLOW: a sphere of air wrapped in a thin
// water film. Looking through the middle you see the sea behind it
// (transparent center); the film is only bright where it curves away
// toward grazing incidence (the rim), plus a small sun glint on the
// sun-facing side of the film. No foam color is painted — all light is
// scattered sun + sky, dimmed by the local wave-slope lighting.
in float vAlpha;
in float vSize;
in float vLight;
in vec2  vSunXY;
out vec4 fragColor;

uniform vec3 uSunColor;

void main()
{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0 || vAlpha <= 0.001) discard;

    float r = sqrt(r2);
    float dome = sqrt(1.0 - r2);              // spherical profile
    vec3 skyAmb = vec3(0.30, 0.36, 0.42);

    // bubble-ness: big rafted foam bubbles are hollow film; small flying
    // droplets are solid water beads
    float bubble = smoothstep(0.05, 0.14, vSize);

    // --- solid water droplet (spray) ---
    vec3 beadCol = (uSunColor * (0.34 + 0.60 * dome)
                    + skyAmb * (0.20 + 0.28 * pow(r, 3.0))) * vLight;
    float beadA = 0.92;

    // --- hollow thin-film bubble (foam) ---
    // film brightens toward the rim (path length through the film grows),
    // center stays almost fully transparent: the dark sea shows through
    float rim = smoothstep(0.45, 0.92, r);
    vec3 filmCol = (uSunColor * (0.30 + 1.05 * rim)
                    + skyAmb * (0.18 + 0.30 * rim)) * vLight;
    // sun glint on the sun-facing side of the film shell
    vec2 gp = c - vSunXY * 0.45;
    filmCol += uSunColor * exp(-dot(gp, gp) * 14.0) * 1.1 * vLight;
    float filmA = mix(0.12, 0.88, rim);       // hollow middle, opaque rim

    vec3 col = mix(beadCol, filmCol, bubble);
    float a = mix(beadA, filmA, bubble);
    a *= smoothstep(1.0, 0.94, r) * vAlpha;
    fragColor = vec4(col, a);
}
