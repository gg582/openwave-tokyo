# Hokusai Sea — Reconstruction of 1830s Tokyo Bay Seascape using CUDA/OpenGL

A C++17 / CUDA / GLSL system reconstructing the physical sea state at the mouth of Tokyo Bay (Uraga Channel / Sagami-nada, **35.2° N, 139.7° E**) — the setting of Katsushika Hokusai's famous print. The simulation combines measured terrain, bathymetry, 1831 solar ephemeris, Mount Fuji altitudinal flora distribution, and Jerlov Coastal II hydro-optics. 

**Note on Philosophy**: This project does not aim for a literal, 100% reproduction of Katsushika Hokusai's woodblock print. Instead, it seeks to recreate a physically plausible "Great Wave" event—the kind of extreme sea state (Draupner rogue wave in a crossing sea) that might have inspired Hokusai's vision of "that day's sea." We aim to capture the atmosphere and the physical power of the ocean that the artist observed.

Running the executable automatically generates four 4K output video files:

| Output File | Pipeline & Features |
|---|---|
| `hokusai_traditional.mp4` | **Control Standard 4K (3840×2160)** — Classic iterative radix-2 FFT + memory-tap unsharp post-fx + FXAA |
| `hokusai_vert30per_traditional.mp4` | **Control 30% Vertically Stretched 4K (3840×2808)** — 30% vertically expanded output of the traditional pipeline |
| `hokusai_complementary.mp4` | **Proposed Standard 4K (3840×2160)** — Warp-shuffle antipodal complementary-pair FFT (`__shfl_xor_sync`) + CUDA inverse-OTF correction + FXAA |
| `hokusai_vert30per_complementary.mp4` | **Proposed 30% Vertically Stretched 4K (3840×2808)** — 30% vertically expanded output of the complementary pipeline |

---

## Technical Details

### 1. Hokusai Composition & Physical Reality
- **Compositional Fidelity**: Recreates the iconic framing using a long-focus/low-height camera frustum ($f=85\text{ mm}$, height $14\text{ m}$) facing Mount Fuji.
- **Draupner Rogue Wave**: Implements the Benjamin-Feir focusing mechanism to generate a 15m+ crest event at $t=15\text{s}$, positioned directly in the foreground.
- **Crossing Seas**: Bimodal JONSWAP spectrum with two systems crossing at $120^\circ$ (McAllister et al., 2019), allowing extreme crest heights.

### 2. Mount Fuji DEM & Ecological Vegetation
- **Triplanar World Mapping**: Eliminates UV stretching on the volcanic flanks.
- **Procedural Volcanic Erosion Gullies**: Computes multi-octave radial FBM erosion channels along the slopes of Mt. Fuji to dark-weather and carve the volcanic soil texture.
- **Wind-Swept High-Resolution Snow Drifts**: Simulates physical snow accumulation with a raised base snow line (2550m) to ensure Mount Fuji's lower forest zones and rock textures remain clearly visible. It clings deeply into upper valleys and gets sheared off by strong NW winds near the summit while clearing from steep cliffs.
- **Ecological Zones**:
  - Lower Woodland (0–1400 m): Dense Cedar / Cypress dark olive-green.
  - Subalpine Coniferous Forest (1400–2100 m): *Abies veitchii* (Veitch fir) deep blue-green.
  - Timberline Boundary (2100–2400 m): *Larix kaempferi* (Japanese Larch) & scoria soil.

### 3. Hydro-Optics & Edo Atmosphere (`shaders/ocean.frag`)
- **1831 Edo Bay Optics (Nutrient-Rich Estuary)**: Spectral attenuation ($\Sigma = (0.280, 0.045, 0.115)\text{ m}^{-1}$) simulating the high microbial/phytoplankton blooms and riverine minerals (CDOM) characteristic of the pre-industrial Edogawa/Sumida river discharges.
- **Procedural Micro-Ripple Glittering**: Generates multi-octave procedural wind ripples flowing along the swell direction. This creates high-fidelity silver sun-glints and specular glittering on close/medium range wave faces.
- **Salinity & Refractive Index**: Adjusted to estuarine 32 PSU levels with a refractive index of $n = 1.338$ reflecting the brackish mixture of the Uraga Channel.
- **Hokusai Prussian Blue**: Subsurface scattering and deep upwelling colors tuned to the *Berliner Blau* palette, shifted slightly toward emerald to reflect the dense marine ecology.
- **Volumetric Exponential Height Fog**: Incorporates low-altitude exponential height fog that dense-hugs the bay and valleys while letting Mt. Fuji's majestic snow peak pierce clearly through the sky.

### 4. Physically Accurate Foam & Spray (Whitecaps & Spindrift)
- **Physical Fluid-Advected Foam Blending**: Actively samples the CUDA-generated Jacobian foam mask in the fragment shader to dictate physical air entrainment. Instead of artificial shapes, the CC0 Foam textures are dynamically advected and stretched by the fluid's horizontal displacement (`uDisp`), mirroring the tearing of real whitecaps under a hemisphere-like diffuse scatter lighting model.
- **Stable Volumetric Spindrift Lighting**: Spray droplets and rafted bubbles are illuminated using a stable, volumetric altitude-based scattering model. This completely eliminates unnatural high-frequency shadow flickering caused by discrete surface normal sampling on flying particles.
- **Crest Breaking & Spindrift**: Restores the physical Jacobian foam generation multiplier (1.8) and exponential foam injection rate (0.45) in the ocean solver. In violent wave-focusing regions, breaking-wave turbulence is dynamically boosted to trigger hundreds of thousands of ballistic sea spray particles (spindrift) ejected at physically-accurate angles based on local wave face slope normals.

---

## Build and Execution

```bash
# Build
make -C hokusai

# Render 60-second (1800 frames) 4K videos for all 4 output targets
./hokusai/hokusai_wave --frames 1800

# Run FFT micro-benchmark
./hokusai/hokusai_wave --bench
```

---

## Used Datasets & Assets

| Asset / Dataset | Source & Details |
|---|---|
| **Mount Fuji DEM** | Mapzen Terrarium PNG encoded DEM (SRTM 30m / USGS NED) |
| **GEBCO Bathymetry** | GEBCO 2024 Grid (15 arc-second resolution) |
| **PBR Materials** | ambientCG CC0 (Foam001, Foam003, Rock063) |
| **Climate Data** | NASA POWER Climatology (35.20° N, 139.70° E) |

---

## References

1. **McAllister, M. L., et al. (2019)**: *Laboratory recreation of the Draupner wave and the role of breaking in crossing seas*, Journal of Fluid Mechanics.
2. **Jerlov, N. G. (1976)**: *Marine Optics*, Elsevier.
3. **Meeus, J. (1998)**: *Astronomical Algorithms*. (1831 Solar Position).
4. **Cook, R. L., & Torrance, K. E. (1982)**: *A Reflectance Model for Computer Graphics*.
