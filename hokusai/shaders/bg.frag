#version 330 core
// ============================================================================
// Physical morning sky, spring 1831. Gradient + forward-scattering glow at
// the astronomically computed solar position for 1831-03-21 07:30 JST;
// cloud coverage follows the NASA POWER March norm for the site (~61%).
// No print-palette stylization — this is the actual morning sky's color.
// ============================================================================
in vec2 vNdc;
out vec4 fragColor;

uniform float uTime;
uniform float uFovY;        // vertical field of view (rad)
uniform float uAspect;
uniform float uCloud;       // cloud coverage 0..1 (NASA POWER climatology)
uniform vec3  uCamDir;      // normalized forward
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uSunDir;      // toward the sun (1831 solar position)
uniform vec3  uSunColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p)
{
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) { v += a * noise(p); p *= 2.03; a *= 0.5; }
    return v;
}

void main()
{
    vec2 ndc = vNdc;
    float tanHalf = tan(uFovY * 0.5);
    vec3 ray = normalize(uCamDir
               + uCamRight * (ndc.x * tanHalf * uAspect)
               + uCamUp    * (ndc.y * tanHalf));

    float elev = ray.y;

    // --- clear morning sky (Rayleigh-style gradient) ---
    vec3 zenith  = vec3(0.130, 0.270, 0.500);
    vec3 horizon = vec3(0.680, 0.700, 0.690);
    // the telephoto frame only spans low elevations, so the blue comes in fast
    vec3 sky = mix(horizon, zenith, smoothstep(-0.02, 0.22, elev));

    // --- sun: forward-scatter glow + disc at the 1831 position ---
    float sd = max(dot(ray, uSunDir), 0.0);
    sky += uSunColor * (0.25 * pow(sd, 48.0) + 0.85 * pow(sd, 900.0));

    // --- clouds: coverage follows the measured climate norm ---
    float az = atan(ray.z, ray.x);
    float cov = fbm(vec2(az * 2.5 + uTime * 0.004, elev * 10.0));
    float thr0 = 1.0 - uCloud * 0.55;
    float cl = smoothstep(thr0, thr0 + 0.25, cov)
             * smoothstep(0.5, 0.06, elev);
    vec3 cloudCol = mix(vec3(0.86, 0.86, 0.84), uSunColor, 0.18);
    sky = mix(sky, cloudCol, cl * 0.6);

    // --- marine haze below the horizon ---
    float sea = smoothstep(0.005, -0.02, elev);
    sky = mix(sky, vec3(0.315, 0.375, 0.420), sea * 0.85);

    fragColor = vec4(sky, 1.0);
}
