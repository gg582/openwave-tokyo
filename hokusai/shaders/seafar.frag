#version 330 core
// ============================================================================
// Far sea plane: the open bay water between the animated wave patch and the
// distant shore, so the coastline never floats. Flat (sub-pixel swell at
// that distance), same water model as the patch: Jerlov body color, water
// Fresnel sky reflection, sun glitter, aerial haze.
// ============================================================================
in vec3 vWorld;
in float vElev;
out vec4 fragColor;

uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uTime;

float sfr_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float sfr_noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(sfr_hash(i), sfr_hash(i + vec2(1, 0)), f.x),
               mix(sfr_hash(i + vec2(0, 1)), sfr_hash(i + vec2(1, 1)), f.x), f.y);
}

void main()
{
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 Hv = normalize(V + L);

    float NoV = clamp(dot(N, V), 1e-4, 1.0);
    float NoL = clamp(dot(N, L), 0.0, 1.0);
    float NoH = clamp(dot(N, Hv), 0.0, 1.0);
    float graze = clamp(1.0 - NoV, 0.0, 1.0);

    // Identical turbid Tokyo Bay water body color as ocean.frag
    vec3 skyRadRaw = mix(uSunColor * vec3(0.12, 0.28, 0.45), uSunColor * vec3(0.22, 0.42, 0.58), graze);
    float skyLum = dot(skyRadRaw, vec3(0.2126, 0.7152, 0.0722));
    vec3 skyRadiance = mix(skyRadRaw, vec3(skyLum), 0.30);
    vec3 deepUpwelling = uSunColor * vec3(0.032, 0.088, 0.082) * (0.60 + 0.40 * NoL);
    vec3 body = deepUpwelling + skyRadiance * 0.13;

    // Physical Fresnel sky reflection (matching ocean.frag F0 = 0.02),
    // broken up by the same wind-glitter modulation as the near patch
    float F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
    vec3 R = reflect(-V, N);
    vec3 skyReflection = mix(skyRadiance * 0.8, skyRadiance * 1.1, pow(clamp(1.0 - R.y, 0.0, 1.0), 2.0));
    float glitter = 0.55 + 0.75 * sfr_noise(vWorld.xz * 0.35 + vec2(uTime * 0.06, -uTime * 0.04))
                         * sfr_noise(vWorld.xz * 0.08 - vec2(uTime * 0.02));
    // The far sea is a matte, sky-lit body of water — not a mirror band:
    // damp the grazing sky reflection so distant water stays close to the
    // near-sea tone
    skyReflection *= 0.55 * glitter;

    // Micro-glint specular
    float glint = pow(NoH, 256.0) * 0.05;

    vec3 color = body + F * skyReflection + uSunColor * glint;

    // Continuous distance fog matching ocean.frag: both converge to the
    // background sky's below-horizon fog tone (one unified sea)
    vec3 horizonFog = vec3(0.42, 0.46, 0.52) * uSunColor * 0.7875;
    float cameraDist = length(uCamPos - vWorld);
    float atmosphericHaze = 1.0 - exp(-cameraDist * 0.00028);
    color = mix(color, horizonFog, atmosphericHaze);

    fragColor = vec4(color, 1.0);
}
