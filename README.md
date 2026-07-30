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
- **Ecological Zones**:
  - Lower Woodland (0–1400 m): Dense Cedar / Cypress dark olive-green.
  - Subalpine Coniferous Forest (1400–2100 m): *Abies veitchii* (Veitch fir) deep blue-green.
  - Timberline Boundary (2100–2400 m): *Larix kaempferi* (Japanese Larch) & scoria soil.

### 3. Hydro-Optics & Edo Atmosphere (`shaders/ocean.frag`)
- **Jerlov Coastal II Optics**: Spectral attenuation ($\Sigma = (0.115, 0.032, 0.058)\text{ m}^{-1}$) matching the estuarine mix of Tokyo Bay.
- **Hokusai Prussian Blue**: Subsurface scattering and deep upwelling colors tuned to the *Berliner Blau* palette and pre-industrial marine aerosols.
- **Sea-Vapor Mist**: Atmospheric scattering simulating the high humidity of the pre-industrial Uraga Channel.

### 4. Physically Accurate Foam & Spray (Whitecaps & Spindrift)
- **Whitecap Foam Blending**: Actively samples the CUDA-generated Jacobian foam mask in the fragment shader. Blends PBR CC0 Foam textures at multiple scales and rotational frames under a hemisphere-like diffuse scatter lighting model.
- **Crest Breaking & Spindrift**: Restores the physical Jacobian foam generation multiplier (1.8) and exponential foam injection rate (0.45) in the ocean solver. In violent wave-focusing regions, breaking-wave turbulence is dynamically boosted to trigger hundreds of thousands of ballistic sea spray particles (spindrift) as soon as waves exceed the Stokes steepness limits.

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
