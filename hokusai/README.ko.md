# 호쿠사이의 바다 — CUDA/OpenGL을 이용한 1830년대 도쿄만 해경 복원

가츠시카 호쿠사이의 유명 판화 배경인 도쿄만 입구(우라가 수로 / 사가미나다, **북위 35.2°, 동경 139.7°**)의 실제 물리적 해상 상태를 복원하는 C++17 / CUDA / GLSL 시스템입니다. 이 시뮬레이션은 실측 지형, 수심 데이터, 1831년 태양력(solar ephemeris), 후지산 고도별 식생 분포, Jerlov Coastal II 해양 광학 모델을 결합하여 구현되었습니다.

**개발 철학에 관한 노트**: 이 프로젝트는 가츠시카 호쿠사이의 목판화를 문자 그대로 100% 재현하는 것을 목표로 하지 않습니다. 대신 교차해(crossing sea)에서 발생하는 드라우프너 이상파랑(Draupner rogue wave)과 같이, 호쿠사이에게 "그날의 바다"에 대한 영감을 주었을 법한 물리적으로 타당한 "거대한 파도" 현상을 재현하고자 합니다. 작가가 목격했을 바다의 분위기와 물리적인 역동성을 담아내는 것이 목표입니다.

실행 파일을 구동하면 다음 네 가지 4K 출력 비디오 파일이 자동으로 생성됩니다:

| 출력 파일 | 파이프라인 및 기능 |
|---|---|
| `hokusai_traditional.mp4` | **대조군 표준 4K (3840×2160)** — 기존의 반복적 radix-2 FFT + 메모리 탭 언샤프 포스트 프로세싱(post-fx) + FXAA |
| `hokusai_vert30per_traditional.mp4` | **대조군 30% 수직 확장 4K (3840×2808)** — 기존 파이프라인의 출력을 수직 방향으로 30% 확장한 결과물 |
| `hokusai_complementary.mp4` | **제안 기법 표준 4K (3840×2160)** — 워프 셔플 대척점 상보쌍 FFT (Warp-shuffle antipodal complementary-pair FFT, `__shfl_xor_sync`) + CUDA 역광학전달함수(inverse-OTF) 보정 + FXAA |
| `hokusai_vert30per_complementary.mp4` | **제안 기법 30% 수직 확장 4K (3840×2808)** — 제안 파이프라인의 출력을 수직 방향으로 30% 확장한 결과물 |

---

## 기술 세부 사항

### 1. 호쿠사이 구도와 물리적 실재
- **구도 충실도**: 후지산을 향하는 망원/저고도 카메라 프러스텀($f=85\text{ mm}$, 고도 $14\text{ m}$)을 사용하여 상징적인 화면 구도를 재현합니다.
- **드라우프너 이상파랑**: 벤자민-페어 포커싱(Benjamin-Feir focusing) 메커니즘을 구현하여 파이프라인 시작 후 $t=15\text{초}$ 시점에 전경에 15m 이상의 파도 마루를 생성합니다.
- **교차해(Crossing Seas)**: 두 파랑 계통이 $120^\circ$ 각도로 교차하는 이봉형 JONSWAP 스펙트럼(McAllister et al., 2019)을 적용하여 극단적인 파고를 구현합니다.

### 2. 후지산 DEM 및 생태학적 식생 분포
- **삼방향 세계 매핑(Triplanar World Mapping)**: 화산 사면에서의 UV 늘어짐 현상을 방지합니다.
- **절차적 수직 침식골 (Volcanic Radial Erosion Gullies)**: 후지산 정상을 중심으로 방사형 수직 침식골을 셰이더 내에서 실시간 연산하여, 화산토 텍스처 사면에 빗물과 화산재 흔적이 깎아낸 정교한 음영 주름을 새겼습니다.
- **바람에 날린 고해상도 만년설 (Wind-Swept Snow Drifts)**: 만년설의 기준 한계선 고도를 상단부(2550m)로 올려, 하부 산림대와 아고산 식생 및 바위 사진 텍스처 데이터가 온전히 보이도록 복구했습니다. 가파른 절벽에는 눈이 쌓이지 못하고 꼭대기 근처의 깊은 침식 수직골 내부로만 눈이 정렬되어 흘러내리도록 만년설 누적 분포를 물리적으로 세밀화했습니다.
- **생태학적 식생대**:
  - 하부 산림대 (0–1400 m): 어두운 올리브 그린 색상의 울창한 삼나무/편백나무림.
  - 아고산대 침엽수림 (1400–2100 m): 짙은 청록색의 구상나무류(*Abies veitchii*, Veitch fir).
  - 삼림한계선 경계부 (2100–2400 m): 일본잎갈나무(*Larix kaempferi*) 및 화산재 토양.

### 3. 수중 광학 및 에도 시대 대기 효과 (`shaders/ocean.frag`)
- **1831년 에도만 수중 광학 (영양염류 풍부)**: 산업화 이전 스미다가와 및 에도가와 강에서 유입되는 풍부한 미생물(식물성 플랑크톤)과 부유 미네랄(CDOM) 농도를 반영하여 분광 감쇄 지수($\Sigma = (0.280, 0.045, 0.115)\text{ m}^{-1}$)를 재조정했습니다.
- **절차적 미세 잔물결과 태양 물비늘 (Micro-Ripple Glittering)**: 파랑 전파 방향에 맞추어 흘러가는 다중 옥타브 미세 바람 물결(procedural wind ripples) 노멀을 합성하여, 태양광 아래에서 은빛 점들로 찬란하게 타오르는 수면 글리터 하이라이트 디테일을 구현했습니다.
- **염도 및 굴절률**: 우라가 수로 특유의 기수역 특성을 반영하여 염도를 32 PSU로 산정하고, 이에 맞춰 바닷물의 굴절률(Refractive Index)을 $n = 1.338$로 적용했습니다.
- **호쿠사이 프러시안 블루**: 프러시안 블루(*Berliner Blau*) 팔레트 기반의 심해 용승색 및 표면하 산란(SSS) 색상에 에도 시대 해양 생태계의 풍부한 에메랄드 톤을 가미하여 정밀 조정했습니다.
- **고도별 기하급수 안개 (Volumetric Exponential Height Fog)**: 저지대 바다와 골짜기 부근에는 해무가 자욱하게 깔리고, 후지산의 만년설 봉우리는 맑은 공기 속에서 하늘을 찌르듯 명확히 드러나는 입체적인 원경 Fog 모델을 구현했습니다.

### 4. 물리적으로 정확한 거품 및 물보라 (Whitecaps & Spindrift)
- **물리 유체 이송(Fluid-Advected) 거품 블렌딩**: 셰이더에서 임의의 기하학적 형태를 그리지 않고 철저히 물과 공기의 물리적 성질에만 의존합니다. CUDA가 연산한 Jacobian 거품 마스크로 공기 유입(Air entrainment) 밀도를 결정하며, 물의 수평 변위 데이터(`uDisp`)를 통해 CC0 거품 텍스처를 동적으로 이송하고 찢어지게 하여 쇄파가 부서지는 현상을 자연스럽게 묘사합니다.
- **안정적인 볼류메트릭 물보라 조명(Volumetric Spindrift Lighting)**: 날아다니는 물보라 입자에 수면의 고주파 이산 노멀을 적용해 발생하던 극심한 깜빡임(flickering) 버그를 원천 차단했습니다. 입자의 고도(altitude)에 기반한 안정적인 볼류메트릭 산란 모델을 적용하여 파도 골짜기에서는 부드럽게 그림자가 지고 공중에선 태양빛을 영롱하게 머금습니다.
- **파도 쇄파 및 물보라(Spindrift)**: 해양 솔버 내에서 물리 기반 Jacobian 거품 생성 승수(1.8) 및 지수 거품 주입률(0.45)을 전격 복구하고, 이상파랑(Rogue wave) 집중 영역에서 쇄파 난류 강도를 대폭 부스팅합니다. 파면의 국소적 경사 법선(wave face slope normal)에 따라 정확한 물리적 궤적으로 비산되도록 개선되어, 파도가 칠 때 세찬 물보라(spindrift) 입자가 역동적으로 사출됩니다.

---

## 빌드 및 실행

```bash
# 빌드
make -C hokusai

# 4개 출력 대상 전체에 대해 60초 분량(1800 프레임)의 4K 비디오 렌더링
./hokusai/hokusai_wave --frames 1800

# FFT 마이크로 벤치마크 실행
./hokusai/hokusai_wave --bench
```

---

## 사용된 데이터셋 및 에셋

| 에셋 / 데이터셋 | 출처 및 세부 정보 |
|---|---|
| **후지산 DEM** | Mapzen Terrarium PNG 인코딩 DEM (SRTM 30m / USGS NED) |
| **GEBCO 수심 데이터** | GEBCO 2024 격자 데이터 (15 arc-second 해상도) |
| **PBR 재질** | ambientCG CC0 (Foam001, Foam003, Rock063) |
| **기후 데이터** | NASA POWER 기후 데이터학 (북위 35.20°, 동경 139.70°) |

---

## 참고 자료

1. **McAllister, M. L., et al. (2019)**: *Laboratory recreation of the Draupner wave and the role of breaking in crossing seas*, Journal of Fluid Mechanics.
2. **Jerlov, N. G. (1976)**: *Marine Optics*, Elsevier.
3. **Meeus, J. (1998)**: *Astronomical Algorithms*. (1831 Solar Position).
4. **Cook, R. L., & Torrance, K. E. (1982)**: *A Reflectance Model for Computer Graphics*.
