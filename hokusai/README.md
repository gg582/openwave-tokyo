# Hokusai Sea — CUDA/OpenGL reconstruction of the 1830s Tokyo Bay seascape

A pure C++17 / CUDA / GLSL system that reconstructs the sea state at the
mouth of Tokyo Bay (Uraga Channel / Sagami-nada side, **35.2 N, 139.7 E**) —
the stage of Katsushika Hokusai's 1830s print — as a photoreal "what did
that sea actually look like back then": real terrain, real bathymetry,
era-appropriate sun and weather, physics-driven waves at the print's scale.
**Not** a moving woodblock print — no print palette, no sculpted wave
shapes; everything large-scale comes from measured data and physics.

ONE scene is advanced per frame and **both kernel pipelines render the same
video** (identical deterministic spectrum seed and clock), so the two MP4s
differ only in the kernels that produced them:

| Output | Pipeline |
|---|---|
| `hokusai_traditional.mp4`   | **Control** — classic iterative radix-2 FFT (all stages through global memory) + unsharp with plain global-memory taps |
| `hokusai_complementary.mp4` | **This work** — warp-shuffle *antipodal complementary-pair* FFT (`__shfl_xor_sync` register exchange) + CUDA warp-shuffle inverse-OTF aberration-correction kernel |

No external APIs or paid libraries: GEBCO-compatible bathymetry
(NetCDF-3 classic, parsed/written in-house), Mapzen Terrarium DEM (real
Mount Fuji), NASA POWER climatology, ambientCG CC0 textures, CUDA, headless
EGL OpenGL 3.3, and the FFmpeg C API with the NVENC H.264 hardware encoder.

---

## Data flow

```
 GEBCO NetCDF ─┐
 (or synthetic ├─> bathymetry H(x,y) ──────────────────────────────┐
  Uraga model)─┘                                                   │
                                                                   v
 JONSWAP spectrum h0(k) ──> time evolution ──> 2D IFFT ──> shoaling/breaking/
 (wind 22 m/s, gamma 3.3)   w²=gk·tanh(kH)    per frame   Jacobian foam   [CUDA]
                                                                   │
                              height / displacement / foam / depth │
                              textures (CUDA→GL interop, no host)  v
                    ┌──────────────────────────────────────────────────┐
                    │ headless EGL OpenGL 3.3                          │
                    │  bg pass: physical morning sky + real Fuji DEM   │
                    │  ocean pass: PBR — Beer-Lambert Jerlov II,       │
                    │  Cook-Torrance GGX, Fresnel F0=0.02, foam        │
                    │  post pass: spherical + chromatic aberration     │
                    └──────────────────────────────────────────────────┘
                                   │ FBO → PBO (async readback)
              ┌────────────────────┴─────────────────────┐
              │ traditional (control)                    │ complementary
              │  GLSL separable Gaussian unsharp         │  CUDA warp-shuffle
              │  (texture taps through memory)           │  inverse-OTF unsharp
              │                                          │  (register exchange)
              └────────────────────┬─────────────────────┘
                                   v
                    RGBA→NV12 kernel ──> NVENC (h264_nvenc,
                    FFmpeg C API, CUDA hw_frames = zero-copy)
                                   v
                              H.264 MP4
```

## 1. GIS bathymetry & shoaling physics (`src/bathymetry.cpp`, `src/ocean.cu`)

- `src/bathymetry.cpp` implements the **NetCDF-3 classic format directly**
  (big-endian header, dims `lat`/`lon`, vars `lat`/`lon`/`elevation`,
  `NC_FLOAT`/`NC_SHORT` with `scale_factor`/`add_offset`). A real GEBCO grid
  loads with `--bathy gebco.nc`; without one, a procedural model of the Uraga
  Channel is synthesized (deep fairway 35–55 m, Miura/Boso banks 5–15 m,
  Sagami-nada slope > 100 m) and written to `data/uraga_synthetic.nc`.
- The wave grid (512², 4 km) is **georeferenced**: every cell knows its
  depth `H(x,y)`.
- Finite-depth dispersion `w² = g k tanh(kH)` per spectral cell,
  **recomputed every frame at the instantaneous tide level**.
- Shoaling/refraction gain map (per frame, CUDA): Green's law
  `Ks = sqrt(cg_deep / cg(H))` and a Snell refraction factor `Kr` from the
  local depth gradient vs. swell direction.
- **Tidal range**: one M2 rise from low to high water
  (±1 m about MSL, Tokyo Bay spring range ~2 m) is time-lapsed over the
  clip. Dispersion, shoaling gain and the breaking criterion all respond to
  `H + tide(t)`, and the surface itself lifts with `uTide` in the vertex
  stage — the tide is told entirely by the WAVES: breaking over the bank
  at low water, a calmer deeper sea at high water.
- **Plunging-breaker collapse**: depth-limited geometric criterion
  (crest > 0.78·(H+tide), McCowan `gamma_b`) plus the **Jacobian
  determinant** `J = (1+l·dDx/dx)(1+l·dDz/dz) − l²·dDx/dz·dDz/dx` of the
  displaced surface; `J` below threshold ⇒ folding crest ⇒ whitecap foam,
  with temporal persistence.
- **Hokusai-scale seas with wind gusts and wave groups**: base wind
  U10 = 25 m/s (spring storm, developed low south of Kanto) with a
  bursty gust envelope (amplitude tracks (U_eff/U)^2, effective gusts
  to ~40 m/s), plus a traveling wave-group envelope at the peak phase
  speed — real sea groupiness — so one dominant set of waves rolls
  through and breaks at mid-clip instead of uniform chop. The breaking bank has an asymmetric profile (gentle seaward
  slope, steep shoreward edge), so crests accelerate over the lip and
  pitch forward — physical plunging breakers that curl into C shapes,
  with displacement choppiness locally amplified as the crest/depth
  ratio approaches the 0.78 H plunging limit (driven into the J < 0
  folding regime at the lip, so the displaced mesh genuinely overturns
  and projects over the trough — the strongest curl a heightfield can
  physically produce), plus a second-order
  bound-wave skewness term (h += 0.9·kp·h|h|, Stokes-type) that
  sharpens crests and flattens troughs so lips curl tightly. A beach
  ramp along the
  south edge of the patch shoals toward a shoreline right next to the
  lens, so the swell curls over the near shore; thin crests get a
  translucent SSS lip glow so the faces read as water.


## 2. Spray droplets — particle extension (`src/spray.cu`, `shaders/spray.*`)

Breaking cells (air fraction above threshold) eject ballistic droplets:
forward at about the wave phase speed, up like the lip, free fall back
into the sea, where they do NOT vanish: landed droplets become floating
foam droplets that ride the wave surface, drift with the residual water
motion and fade over a 2–5 s lifetime — every water droplet tracked
individually (dead / flying / floating state machine). Two of the kernels ship in BOTH variants for
the A/B benchmark: emitter scan (per-cell global atomicAdd vs
warp-ballot + shuffle prefix scan) and grid projection (naive atomics
vs `__match_any` warp-aggregated atomics). Droplets render as GL point
sprites lit by the same environment light. Outputs are bit-identical
between variants (Q16.16 deposits, host-sorted emitter list).

## 3. Barrel lip ribbon (plunging-jet model) (`src/renderer.cpp` `updateBarrel`, `shaders/barrel.*`)

Surfers' tube is real physics: at a plunging break the lip is ejected at
about the wave phase speed and curls over on a ballistic trajectory
(Longuet-Higgins & Cokelet). Every frame we find the actual breaking
crest line in the simulated height field, launch a jet cross-section
ballistically (`v0 ≈ 1.1·cp`, `vy0 = 0.35·sqrt(g·h)`, free fall), and
build a ribbon surface along the coherent crest (median-filtered, so no
stitched artifacts). The result is a true overturning "C" surface that a
heightfield alone cannot represent — generated from the simulated wave,
never a canned shape.

## 4. Coastal PBR (Jerlov Type II) (`shaders/ocean.frag`)

- **Path-length Beer-Lambert transparency** with per-channel
  absorption `sigma = vec3(0.08, 0.03, 0.01)`: thin near-surface water
  (lips, crests, shallow shore) transmits, transparency collapses as
  the underwater path grows (deeper columns, grazing rays); sandy
  bottom shows through thin shallow columns viewed steeply.
- **Cook-Torrance BRDF**: GGX distribution, Smith geometry, Schlick Fresnel
  with water **F0 = 0.02**; foam regions switch to the photographed
  roughness map.
- **Air-entrainment foam** (an actual equation, not a white overlay):
  breaking injects air at rate q (Jacobian folding + depth-limited
  plunging), the air fraction A decays as bubbles rise/dissolve
  (`A = 0.9·A + 0.8·q`, capped at 1.5). Rendering follows the physics and
  the reference print (Met Museum DP141063, public domain, consulted):
  a solid bright lip core, streaky trails draining down the face along
  propagation, spray droplets — brightness tracks A (fresh white lip,
  drained gray wash), with high-scatter shading instead of flat white.
  On top of that, each individual bubble is rendered as a small shaded
  sphere (world-space cellular field with dome glint and per-bubble
  lifetime), so the foam reads as round bubbles up close. The seafloor
  also shows through thin shallow columns: photographed bottom detail
  shaded by the actual depth gradient. Foam color is not painted white:
  bubbles multiply-scatter the actual scene light (the 1831 sun plus sky
  ambient), with the optical depth of the aerated layer setting the
  opacity.

### CC0 material assets (downloaded at setup, see `assets/`)

Photographed PBR maps from [ambientCG](https://ambientcg.com/) (CC0) are
layered onto the spectral surface for close-up realism:

| Asset | Maps used | Where applied |
|---|---|---|
| Foam 001 (`assets/Foam001/`) | Color, NormalGL | two octaves of drifting micro-ripple normals over the whole surface; whitecap albedo + relief inside foam regions |
| Foam 003 (`assets/Foam003/`) | Opacity | broken, lacy whitecap coverage shapes driven by the CUDA foam mask |
| Rock 063 (`assets/Rock063/`) | Color | photographic rock on the real Fuji terrain flanks, natural warm-gray grade |
| Terrarium DEM (`assets/fuji_dem.*`) | elevation | real Mount Fuji terrain mesh (Mapzen terrain tiles, AWS Open Data, CC-BY — USGS/NASA/NGA/GSI sources) |

The simulation still provides all large-scale shape and motion; the
photographed maps only add sub-grid surface detail.

## 5. The Hokusai lens & real-time aberration correction

**Composition (`src/renderer.cpp`)**: a long-focus frustum (`fov = 26 deg`)
at 16 m above the swell, aimed WNW (bearing 289 deg) toward the real Fuji —
the actual viewpoint geometry of the print. The sky is physical: morning
gradient, forward-scatter glow at the computed 1831 solar position, cloud
coverage from the measured climate norm.

**Era sun & weather (`src/climate.cpp`, `assets/climate.json`)**: the solar
position is computed from Earth's orbital elements AT THE 1831 EPOCH —
mean longitude/anomaly, eccentricity, obliquity, equation of center and
GMST via the standard Meeus series in double precision — giving
1831-03-21 07:30 JST: elevation 20.3°, azimuth 105.4° (ESE morning sun),
Sun-Earth distance 0.99669 AU. The sunlight COLOR is computed from the
same elevation: Beer-Bouguer extinction through air mass 2.86
(Kasten-Young) with per-channel Rayleigh + marine-aerosol coefficients
→ (1.034, 0.944, 0.764), used for every lit surface including foam.
Weather comes
from NASA POWER climatology for the site (March: 4.11 kWh/m²/day solar,
61 % cloud, 11.1 °C, 6.0 m/s wind).

**Real terrain (`src/terrain.cpp`, `shaders/terrain.*`)**: the mountains
are NOT an analytic cone and NOT a floating tile board — the mesh is a real
elevation field from **Mapzen Terrarium DEM tiles** (AWS Open Data, CC-BY),
48 z11 tiles stitched by `assets/fetch_fuji_dem.sh` covering the whole view
corridor (lon 138.52–139.92 E, lat 34.89–35.75 N: the Miura/Izu shoreline
all the way to Mt. Fuji, whose summit pixel decodes to 3716 m ≈ the real
3776 m). Every vertex sits at its TRUE geographic position relative to the
wave patch, so Fuji stands at its real ~91 km distance and the shoreline
grounds the frame. A **far sea plane** (`shaders/seafar.frag`) fills the
bay between the wave patch and the coast.
Playback timing is physically locked: one playback second always advances
the simulation by `--timescale` (3.0) sim-seconds at `--fps` (30) —
rendering throughput never affects animation speed.

**Aberration (`shaders/aberration.frag`)**: field-dependent spherical blur
(`~ r⁴`, sharp center, soft corners) + lateral chromatic aberration
(`~ r²`, purple/green fringing), emulating a period spherical singlet.

**Correction — the two paths compared:**

- *Traditional control* (`src/postfx.cu`): the SAME CUDA unsharp, but
  every tap is a plain global-memory fetch — no register exchange.
- *Complementary* (`src/postfx.cu`): a CUDA **inverse-OTF** kernel —
  1. inverse lateral-chromatic re-alignment (R/B bilinear re-sampling at the
     inverse radial displacement),
  2. 5-tap Gaussian unsharp whose **horizontal neighbor exchange runs
     entirely in registers** via `__shfl_up/down_sync` (antipodal lane-pair
     exchange, ~1 cycle, no shared memory, no extra global traffic),
  3. field-dependent gain `g(r) = amount·(1 + alpha·r⁴)` matching the `r⁴`
     growth of the aberration blur — edge contrast restored exactly where
     the lens destroyed it.

## 4. Antipodal complementary-pair FFT (`src/fft_gpu.cu`)

Both pipelines compute the same radix-2 DIT FFT; only the operand transport
differs:

- **Complementary**: lanes `T_i` / `T_{i+16}` (and in general `T_i` /
  `T_{i^mask}`) exchange `A` and `B·W` through `__shfl_xor_sync` at register
  level. Primary lanes compute `A + B·W`, secondary lanes `A − B·W`; the role
  is selected **branch-free** from the laneId bit via an IEEE-754 sign-bit
  XOR (`sec << __clz(mask)`), so the warp never diverges. Stages with span
  1–16 (5 of log2N stages) run entirely in registers; only spans ≥ 32 fall
  back to global-memory butterfly passes. DRAM passes drop from `log2N` to
  `log2N − 4`. (Standalone demo: `../complementary_fft.cu`.)
- **Traditional (control)**: bit-reversed copy + every stage as a full
  global-memory pass — the structure a generic library plan uses.

Numerical equivalence of the two was verified against a CPU DFT
(max error ~1e-9 at 256²).

## 5. Zero-copy MP4 encoding (`src/encoder.cpp`, `src/postfx.cu`)

```
GL FBO → PBO (glReadPixels, async)
      → cudaGraphicsGLRegisterBuffer mapping (device pointer)
      → [complementary: correction kernel in place]
      → RGBA→NV12 kernel writing DIRECTLY into FFmpeg AVFrame CUDA planes
      → h264_nvenc (NVENC) → H.264 MP4
```

Pixels never leave the GPU. If NVENC is unavailable the encoder falls back
to libx264 (same H.264/MP4 contract).

## Build & run

```bash
make                       # nvcc + EGL/GL + FFmpeg dev libs
./hokusai_wave --mode complementary --frames 240 --out hokusai_complementary.mp4
./hokusai_wave --mode traditional  --frames 240 --out hokusai_traditional.mp4
./hokusai_wave --bench      # FFT micro-benchmarks (256/512/1024)
./hokusai_wave --bathy gebco.nc --mode complementary --out gebco.mp4
```

Options: `--frames N` (default 240), `--fps N` (default 30 — playback rate
is locked to this; rendering speed never changes it), `--timescale S`
(default 0.9 sim-seconds per playback second), `--width/--height`
(default 1280×720), `--bathy FILE.nc`, `--shaders DIR`, `--bench`.

## Repository layout

```
hokusai/
├── Makefile
├── README.md / README.ko.md        (English / Korean)
├── benchmark.md / benchmark.ko.md  (measured results)
├── shaders/
│   ├── bg.vert / bg.frag           physical morning sky, sun glow, clouds
│   ├── ocean.vert / ocean.frag     displaced mesh + Jerlov II PBR
│   ├── spray.vert / spray.frag     droplet point sprites
│   ├── barrel.vert / barrel.frag   plunging-jet ribbon
│   ├── seafar.frag                 far sea plane (bay to the horizon)
│   ├── aberration.frag             spherical + chromatic aberration
│   └── unsharp.frag                traditional GLSL unsharp (control)
├── src/
│   ├── main.cpp                    pipeline driver + per-stage timers
│   ├── bathymetry.h/.cpp           NetCDF-3 classic I/O + synthetic Uraga
│   ├── ocean.h/.cu                 JONSWAP, dispersion, shoaling, breaking,
│   │                             Jacobian foam
│   ├── fft_gpu.h/.cu               complementary warp-shuffle FFT +
│   │                             traditional control + 2D driver + bench
│   ├── renderer.h/.cpp             headless EGL GL3.3, FBO chain, PBO,
│   │                             CUDA-GL interop
│   ├── texture_load.h/.cpp         minimal JPEG loader (libjpeg)
│   ├── terrain.h/.cpp              Terrarium DEM → real Fuji mesh
│   ├── climate.h/.cpp              1831 solar ephemeris + NASA POWER weather
│   ├── postfx.h/.cu                warp-shuffle inverse-OTF correction,
│   │                             PBO mapping, RGBA→NV12
│   └── encoder.h/.cpp              FFmpeg C API, NVENC CUDA hw_frames,
│                                 libx264 fallback
├── assets/                         CC0 material maps (ambientCG):
│                                   Foam001, Foam003, Rock063
└── data/uraga_synthetic.nc         generated on first run
```

## Requirements

- NVIDIA GPU + CUDA toolkit (built with 13.1, sm_86)
- EGL 1.5 + OpenGL 3.3 (headless; no X11 needed)
- FFmpeg development libraries (`libavformat`, `libavcodec`, `avutil`)
  with `h264_nvenc` for the zero-copy path
