# 호쿠사이 씨 — 1830년대 도쿄만 입구 해경의 CUDA/OpenGL 재건

1830년대 가츠시카 호쿠사이의 판화 배경인 도쿄만 입구(우라가 수도/사가미나다 측, **북위 35.2°, 동경 139.7°**)의 바다를, "그 시절 그 바다가 실제로 어땠을지"라는 질문에 답하는 실사 재현으로 순수 C++17 / CUDA / GLSL만으로 시뮬레이션합니다 — 실측 지형, 실측 수심, 당시 태양·날씨, 물리 기반 파도. **움직이는 판화가 아닙니다**: 판화 팔레트도, 조각한 파도 형태도 없습니다.

매 프레임 하나의 장면을 진행시키고, **두 커널 파이프라인이 동일한 영상**을 렌더링합니다(결정론적 스펙트럼 시드와 클럭이 동일) — 두 MP4는 커널만 다릅니다:

| 출력 | 파이프라인 |
|---|---|
| `hokusai_traditional.mp4`   | **대조군** — 전통적 반복 radix-2 FFT(모든 스테이지가 글로벌 메모리 경유) + 글로벌 메모리 탭 언샤프 |
| `hokusai_complementary.mp4` | **본 커널** — warp-shuffle **대척 보수쌍** FFT(`__shfl_xor_sync` 레지스터 교환) + CUDA warp-shuffle 역OTF 수차 보정 커널 |

외부 API·유료 라이브러리 없음: GEBCO 호환 수심 데이터(NetCDF-3 classic 자체 파서), Mapzen Terrarium DEM(실측 후지산), NASA POWER 기후, ambientCG CC0 텍스처, CUDA, 헤드리스 EGL OpenGL 3.3, FFmpeg C API + NVENC H.264 하드웨어 인코더.

## 데이터 흐름

```
 GEBCO NetCDF ─┐
 (또는 합성   ├─> 수심 H(x,y) ─────────────────────────────────┐
  우라가 모델)─┘                                                │
                                                                v
 JONSWAP 스펙트럼 h0(k) ─> 시간 진화 ─> 2D IFFT ─> 쇼어링/쇄파/  [CUDA]
 (풍속 22m/s, γ=3.3)      w²=gk·tanh(kH)   프레임별   야코비안 폼
                                                                │
                        높이/변위/폼/수심 텍스처(CUDA→GL interop) v
              ┌────────────────────────────────────────────────────┐
              │ 헤드리스 EGL OpenGL 3.3                            │
              │  배경: 물리적 아침 하늘 + 실측 후지산 DEM          │
              │  해양: PBR — Beer-Lambert Jerlov II,               │
              │  Cook-Torrance GGX, Fresnel F0=0.02, 폼            │
              │  후처리: 구면 + 색 수차                            │
              └────────────────────────────────────────────────────┘
                          FBO → PBO (비동기 판독)
              ┌───────────────────┴────────────────────┐
              │ 전통 (대조군)                          │ 보수쌍
              │  언샤프: 글로벌 메모리 탭             │  언샤프: warp-shuffle
              │  (모든 탭이 메모리 경유)               │  역OTF 언샤프 (레지스터 교환)
              └───────────────────┬────────────────────┘
                                  v
                  RGBA→NV12 커널 ─> NVENC (h264_nvenc,
                  FFmpeg C API, CUDA hw_frames = zero-copy)
                                  v
                             H.264 MP4
```

## 1. GIS 수심 연동 및 천해 물리 (`src/bathymetry.cpp`, `src/ocean.cu`)

- NetCDF-3 classic 포맷을 **직접 구현**(빅엔디안 헤더, `lat`/`lon` 차원,
  `elevation` 변수, `NC_FLOAT`/`NC_SHORT` + `scale_factor`/`add_offset`).
  실제 GEBCO 그리드는 `--bathy gebco.nc`로 투입 가능. 없으면 우라가 수도
  절차적 모델(심해 수로 35–55m, 미우라/보소 천대 5–15m, 사가미나다
  대륙사면 100m+)을 생성해 `data/uraga_synthetic.nc`로 저장.
- 파도 그리드(512², 4km)는 **지리 참조됨**: 모든 셀이 수심 `H(x,y)`를 가짐.
- 유한수심 분산 관계 `w² = g k tanh(kH)` (스펙트럴 셀별) — **매 프레임
  순간 조위로 재계산**.
- 쇼어링/굴절 게인 맵(프레임별, CUDA): 그린의 법칙
  `Ks = sqrt(cg_deep / cg(H))` + 국소 수심 기울기와 스웰 방향으로부터의
  Snell 굴절 계수 `Kr`.
- **조석간만의 차**: M2 반일주조의 간조→만조(평균수면 ±1m, 도쿄만
  최대 조차 ~2m)를 클립 전체에 타임랩스로 재현. 분산·쇼어링 게인·쇄파
  기준이 모두 `H + tide(t)`에 반응하고, 버텍스 스테이지의 `uTide`로
  수면 자체가 승강 — 조석은 오직 파도로만 표현됩니다: 간조에는 천대
  위 쇄파, 만조에는 깊어져 잔잔한 바다.
- **쇄파 붕괴**: 수심 제한 기하학적 기준(정상부 > 0.78·(H+tide),
  McCowan γ_b) + 변위 표면의 **야코비안 행렬식** `J`가 임계 이하 ⇒
  파면 접힘 ⇒ 백파 폼(시간적 지속성 포함).
- **돌풍과 파장 그룹을 동반한 호쿠사이 스케일 해황**: 기본풍
  U10 = 25m/s(춘계 발달 저기압)에 돌풍 엔벨로프(진폭이 (U_eff/U)²로
  추종, 유효 돌풍 ~40m/s)와 피크 위상속도로 이동하는 파장 그룹
  엔벨로프(실제 바다의 그룹성)를 얹어, 균일한 잔파도 대신 지배적인
  하나의 파도 세트가 클립 중간에 밀려와 부서집니다. 쇄파 천대는 비대칭(바다 쪽 완만, 해안 쪽 급경사)이라
  정상부가 립 위에서 가속되어 앞으로 넘어가는 물리적 플런징 브레이커
  — 정상/수심비가 0.78H 한계에 가까워질수록 전방 초피니스를 접힘
  영역(J < 0)까지 증폭해 변위 메쉬가 실제로 뒤집혀 립이 골짜기 위로
  넘어가며(높이장 모델이 만들 수 있는 최대 컬), 2차 결합파 스큐네스 항(h += 0.9·kp·h|h|, Stokes형)이
  정상을 뾰족하게 깎아 립이 타이트하게 말립니다. 패치 남쪽
  가장자리의 비치 램프가
  렌즈 바로 앞 해안선까지 수심을 낮춰 스웰이 근해안에서 휘어 치며,
  얇은 정상부에 SSS 립 투과광을 넣어 파도면이 물로 읽힙니다.

## 2. 스프레이 입자 — 입자 기반 확장 (`src/spray.cu`, `shaders/spray.*`)

공기분율이 임계를 넘는 쇄파 셀이 탄도 물방울을 사출합니다: 위상속도로
전방 사출, 립처럼 상승 후 자유낙하 — 그리고 착수 후에도 소멸하지
않습니다: 떨어진 물방울은 부유 거품 입자가 되어 파도면을 타고 표류하며
2–5초 수명 동안 페이드됩니다(dead / flying / floating 상태 기계로 물방울
하나하나를 추적) — 높이장이 담을 수 없는 뾰족하고 거친 비산. 두 커널은 A/B 벤치를 위해
두 변형을 갖습니다: 발사체 스캔(셀별 글로벌 atomicAdd 대 warp-ballot +
shuffle 프리픽스 스캔)과 그리드 투영(naive atomic 대 `__match_any`
warp 집계 atomic). 입자는 동일 환경광으로 빛나는 GL 포인트
스프라이트로 렌더링되며, 두 변형의 출력은 비트 동일합니다(Q16.16
고정소수 퇴적, 호스트 정렬 발사체 목록).

## 3. 배럴 립 리본 (플런징 제트 모델) (`src/renderer.cpp` `updateBarrel`, `shaders/barrel.*`)

서퍼가 타는 튜브는 실제 물리입니다: 플런징 쇄파에서 립은 위상속도로
사출되어 탄도 궤적으로 말립니다(Longuet-Higgins & Cokelet). 매 프레임
시뮬레이션된 높이장에서 실제 쇄파 크레스트 라인을 찾아 탄도 제트 단면을
발사(`v0 ≈ 1.1·cp`, `vy0 = 0.35·sqrt(g·h)`, 자유낙하)하고, 일관된 크레스트
라인(중앙값 필터)을 따라 리본을 생성합니다 — 높이장만으로는 표현할 수
없는 진짜 오버터닝 "C" 표면을, 조각이 아닌 시뮬레이션 파도로부터 만듭니다.

## 4. 연안 PBR (Jerlov Type II) (`shaders/ocean.frag`)

- **광학 경로 길이 Beer-Lambert 투명도**: 채널별 흡수 계수
  `sigma = vec3(0.08, 0.03, 0.01)` — 얇은 표층수(립, 정상부, 천해)는
  투과하고 수중 경로가 길어질수록(깊은 물기둥, 그레이징) 불투명해지며,
  얕은 곳을 낮게 볼 때는 모래 바닥이 비칩니다.
- **Cook-Torrance BRDF**: GGX 분포, Smith 기하, Schlick Fresnel,
  물의 **F0 = 0.02**.
- **공기 연행 폼**(흰색 덧칠이 아닌 산식): 쇄파(야코비안 접힘 + 수심
  제한 플런징)이 공기를 q율로 주입하고 공기분율 A는 기포 상승·용해로
  감쇠(`A = 0.9A + 0.8q`, 상한 1.5). 렌더링은 참고 원화(Met Museum
  DP141063, 퍼블릭 도메인)를 따라 립 코어 + 파면 스트릭 + 비산 물방울의
  3층 구조이며 폼 색은 흰색 덧칠이 아닙니다: 기포가 실제 장면 광
  (1831년 태양 + 천공 앰비언트)을 다중 산란하고, 공기화 층의 광학
  깊이가 불투명도를 결정합니다. 여기에 개별 기포 하나하나를 작은
  구형(돔 하이라이트 + 가장자리림, 기포별 수명 주기)으로 렌더링해
  거품이 동글동글하게 보입니다. 얕은 물 아래로는 실측 수심 구배로
  음영 처리된 해저 지형도 비칩니다.

### CC0 머터리얼 에셋 (`assets/`)

[ambientCG](https://ambientcg.com/)(CC0)의 실측 PBR 맵을 스펙트럴 파면 위에
합성해 근경 사실감을 높였습니다:

| 에셋 | 사용 맵 | 적용 |
|---|---|---|
| Foam 001 (`assets/Foam001/`) | Color, NormalGL | 전 수면 드리프트되는 미세 리플 노멀 2옥타브, 백파 영역 알베도+릴리프 |
| Foam 003 (`assets/Foam003/`) | Opacity | CUDA 폼 마스크를 따르는 레이스 모양 백파 커버리지 |
| Rock 063 (`assets/Rock063/`) | Color | 실측 후지산 지형 측면의 실사 암석(자연 웜그레이) |
| Terrarium DEM (`assets/fuji_dem.*`) | 고도 | 실측 후지산 지형 메쉬(Mapzen terrain tiles, AWS 오픈 데이터, CC-BY — USGS/NASA/NGA/GSI 소스) |

대규모 형상·울렁임은 전부 시뮬레이션이 만들고, 실사 맵은 서브그리드 표면
디테일만 보강합니다.

## 5. 호쿠사이 렌즈와 실시간 수차 개선

**구도 (`src/renderer.cpp`)**: 초장초점 프러스텀(`fov = 26°`), 스웰 위
16m, 실제 후지산 방위(WNW 289°) 조준 — 판화의 실제 시점 기하. 하늘은
물리적 아침 하늘(1831년 태양 위치의 순방향 산란 광휘, 기후 norm의
구름량).

**당시 태양·날씨 (`src/climate.cpp`, `assets/climate.json`)**: 태양 위치는
1831년 시점의 지구 궤도 요소(평균 황경/근점이각, 이심률, 황도경사,
중심차, GMST)를 Meeus 급수로 double 정밀도 역산 — 1831-03-21 07:30
JST 기준 고도 20.3°, 방위각 105.4°(ESE 아침 태양), 태양-지구 거리
0.99669 AU. 태양빛 색상도 같은 고도에서 계산합니다: 기단 질량 2.86
(Kasten-Young)에 대한 Beer-Bouguer 소광(채널별 Rayleigh + 해양 에어로졸)
→ (1.034, 0.944, 0.764)을 폼을 포함한 모든 조명 표면에 적용.
날씨는 NASA POWER 기후
norm(3월: 일사 4.11 kWh/m²/day, 구름 61%, 기온 11.1°C, 풍속
6.0m/s)을 사용.

**실측 후지산 지형 (`src/terrain.cpp`, `shaders/terrain.*`)**: 산은
해석적 원뿔이 **아닙니다** — `assets/fetch_fuji_dem.sh`가 남려받는
**Mapzen Terrarium DEM 타일**(AWS 오픈 데이터, CC-BY)로 만든 실측 고도
메쉬입니다(z11 타일, 35.3606N 138.7274E 주변, Terrarium 인코딩; 정상부
픽셀 3744m ≈ 실측 3776m). 384×256 메쉬를 실제 방위 ~70km 지점에 배치하고
CC0 암석 텍스처·고도/경사 기반 설피·대기 원근감으로 셰이딩합니다.
파도 패치와 해안 사이의 만은 **원해 평면**(`shaders/seafar.frag`)으로
메워 해안선이 붕 뜨지 않게 했습니다. 재생 시간은 물리적으로 고정됩니다:
재생 1초 = 정확히 `--timescale`(3.0) 시뮬레이션 초(`--fps` 30) —
렌더링 처리량은 애니메이션 속도에 전혀 영향을 주지 않습니다.

**수차 (`shaders/aberration.frag`)**: 시야각 의존 구면수차 블러(`~ r⁴`,
중심 선명·주변부 흐림) + 횡색수차(`~ r²`, 퍼플/그린 프린징).

**보정 — 두 경로 비교:**

- *전통 대조군* (`src/postfx.cu`): 동일한 CUDA 언샤프이나 모든 탭이
  평범한 글로벌 메모리 페치 — 레지스터 교환 없음.
- *보수쌍* (`src/postfx.cu`): CUDA **역광학전달함수(역OTF)** 커널 —
  1. 횡색수차 역정렬(R/B 채널을 역방향 반경 변위로 쌍선형 재샘플링),
  2. 수평 이웃 교환을 **전부 레지스터에서** `__shfl_up/down_sync`로
     처리하는 5탭 가우시안 언샤프(대척 레인쌍 교환, ~1사이클,
     공유 메모리·추가 글로벌 트래픽 0),
  3. 시야 의존 게인 `g(r) = amount·(1 + alpha·r⁴)` — 수차 블러의 r⁴
     성장과 동형으로, 렌즈가 묻은 주변부 에지 대비를 정확히 복원.

## 4. 대척 보수쌍 FFT (`src/fft_gpu.cu`)

두 파이프라인 모두 같은 radix-2 DIT FFT를 계산하며, 피연산자 운송 방식만
다릅니다:

- **보수쌍**: 레인 `T_i`/`T_{i+16}`(일반적으로 `T_i`/`T_{i^mask}`)이
  `A`와 `B·W`를 `__shfl_xor_sync`로 레지스터 레벨 교환. Primary 레인은
  `A + B·W`, Secondary 레인은 `A − B·W`를 계산하고, 역할은 laneId 비트를
  IEEE-754 부호 비트 XOR(`sec << __clz(mask)`)로 변환해 **무분기**로
  선택(워프 다이버전스 없음). span 1–16의 5개 스테이지는 레지스터에서
  완결되고 span ≥ 32만 글로벌 메모리 패스로 남음. DRAM 패스가 log2N에서
  log2N−4로 감소. (단독 데모: `../complementary_fft.cu`)
- **전통 (대조군)**: 비트 반전 복사 + 모든 스테이지가 글로벌 메모리
  패스(범용 라이브러리 플랜의 구조).

두 구현의 수치 동등성은 CPU DFT 대비 최대 오차 ~1e-9 (256²)로 검증됨.

## 5. Zero-copy MP4 인코딩 (`src/encoder.cpp`, `src/postfx.cu`)

```
GL FBO → PBO (glReadPixels, 비동기)
      → cudaGraphicsGLRegisterBuffer 매핑 (디바이스 포인터)
      → [보수쌍: 보정 커널 in-place]
      → RGBA→NV12 커널이 FFmpeg AVFrame CUDA 평면에 직접 기록
      → h264_nvenc (NVENC) → H.264 MP4
```

픽셀이 GPU를 떠나지 않습니다. NVENC이 없으면 libx264로 폴백(동일한
H.264/MP4 출력).

## 빌드 및 실행

```bash
make                       # nvcc + EGL/GL + FFmpeg 개발 라이브러리 필요
./hokusai_wave --mode complementary --frames 240 --out hokusai_complementary.mp4
./hokusai_wave --mode traditional  --frames 240 --out hokusai_traditional.mp4
./hokusai_wave --bench      # FFT 마이크로 벤치마크 (256/512/1024)
./hokusai_wave --bathy gebco.nc --mode complementary --out gebco.mp4
```

옵션: `--frames N`(기본 240), `--fps N`(기본 30 — 재생 레이트는 이 값으로
고정되며 렌더링 속도가 바꾸지 못함), `--timescale S`(재생 1초당
시뮬레이션 초, 기본 3.0), `--width/--height`(기본 1280×720),
`--bathy FILE.nc`, `--shaders DIR`, `--bench`.

## 저장소 구성

```
hokusai/
├── Makefile
├── README.md / README.ko.md        (영문 / 한글)
├── benchmark.md / benchmark.ko.md  (측정 결과)
├── shaders/
│   ├── bg.vert / bg.frag           물리적 아침 하늘, 태양 광휘, 구름
│   ├── ocean.vert / ocean.frag     변위 메쉬 + Jerlov II PBR
│   ├── seafar.frag                 원해 평면 (수평선까지의 만)
│   ├── aberration.frag             구면 + 색 수차
│   └── unsharp.frag                전통 GLSL 언샤프 (대조군)
├── src/
│   ├── main.cpp                    파이프라인 드라이버 + 단계별 타이머
│   ├── bathymetry.h/.cpp           NetCDF-3 classic I/O + 합성 우라가
│   ├── ocean.h/.cu                 JONSWAP, 분산, 쇼어링, 쇄파, 야코비안 폼
│   ├── fft_gpu.h/.cu               보수쌍 warp-shuffle FFT + 전통 대조군
│   │                             + 2D 드라이버 + 벤치
│   ├── renderer.h/.cpp             헤드리스 EGL GL3.3, FBO 체인, PBO,
│   │                             CUDA-GL interop
│   ├── postfx.h/.cu                warp-shuffle 역OTF 보정, PBO 매핑,
│   │                             RGBA→NV12
│   ├── texture_load.h/.cpp         최소 JPEG 로더 (libjpeg)
│   ├── terrain.h/.cpp              Terrarium DEM → 실측 후지산 메쉬
│   ├── climate.h/.cpp              1831년 태양 역천 + NASA POWER 날씨
│   └── encoder.h/.cpp              FFmpeg C API, NVENC CUDA hw_frames,
│                                 libx264 폴백
├── assets/                         CC0 머터리얼 맵 (ambientCG):
│                                   Foam001, Foam003, Rock063
└── data/uraga_synthetic.nc         첫 실행 시 생성
```

## 요구 사항

- NVIDIA GPU + CUDA 툴킷 (13.1, sm_86으로 빌드)
- EGL 1.5 + OpenGL 3.3 (헤드리스, X11 불필요)
- FFmpeg 개발 라이브러리(`libavformat`, `libavcodec`, `avutil`),
  zero-copy 경로에는 `h264_nvenc`
