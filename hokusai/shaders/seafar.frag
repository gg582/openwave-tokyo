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

    // Identical turbid Tokyo Bay water body color as ocean.frag (Beer-Lambert model)
    vec3 skyRadRaw = mix(uSunColor * vec3(0.12, 0.28, 0.45), uSunColor * vec3(0.22, 0.42, 0.58), graze);
    float skyLum = dot(skyRadRaw, vec3(0.2126, 0.7152, 0.0722));
    vec3 skyRadiance = mix(skyRadRaw, vec3(skyLum), 0.30);
    
    const vec3 EdoBay_Sigma = vec3(0.280, 0.045, 0.115);
    float pathLen = 45.0 * (1.0 / max(NoV, 0.08) + 1.0 / max(NoL, 0.08));
    vec3 waterTransmittance = exp(-EdoBay_Sigma * pathLen);
    vec3 backscatterCoeff = vec3(0.015, 0.098, 0.145) * 0.35; 
    vec3 volumeScattering = backscatterCoeff * (1.0 - waterTransmittance) / (EdoBay_Sigma + 1e-4);
    vec3 deepUpwelling = uSunColor * volumeScattering * (0.50 + 0.50 * NoL);
    vec3 body = deepUpwelling + skyRadiance * 0.08;

    // Physical Fresnel sky reflection (matching ocean.frag F0 = 0.02),
    // broken up by the same wind-glitter modulation as the near patch
    float F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
    vec3 R = reflect(-V, N);
    vec3 skyReflection = mix(skyRadiance * 0.8, skyRadiance * 1.1, pow(clamp(1.0 - R.y, 0.0, 1.0), 2.0));
    float glitter = 0.55 + 0.75 * sfr_noise(vWorld.xz * 0.35 + vec2(uTime * 0.06, -uTime * 0.04))
                         * sfr_noise(vWorld.xz * 0.08 - vec2(uTime * 0.02));
    skyReflection *= 0.55 * glitter;

    // Anisotropic spec mapping for far plane wind-aligned glints
    float roughX = 0.12; 
    float roughY = 0.28; 
    vec3 T_aniso = vec3(0.325, 0.0, -0.946); 
    vec3 B_aniso = cross(N, T_aniso);
    float XoH = dot(T_aniso, Hv);
    float YoH = dot(B_aniso, Hv);
    float d_denom = XoH*XoH / (roughX*roughX) + YoH*YoH / (roughY*roughY) + NoH*NoH;
    float D_aniso = 1.0 / (3.14159265 * roughX * roughY * d_denom * d_denom + 1e-5);
    float glint = D_aniso * 0.0035;

    vec3 color = body + F * skyReflection + uSunColor * glint;

    // Continuous distance fog matching ocean.frag: both converge to the
    // background sky's below-horizon fog tone (one unified sea)
    vec3 horizonFog = vec3(0.42, 0.46, 0.52) * uSunColor * 0.7875;
    float cameraDist = length(uCamPos - vWorld);
    float atmosphericHaze = 1.0 - exp(-cameraDist * 0.00028);
    color = mix(color, horizonFog, atmosphericHaze);

    fragColor = vec4(color, 1.0);
}
