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
- **생태학적 식생대**:
  - 하부 산림대 (0–1400 m): 어두운 올리브 그린 색상의 울창한 삼나무/편백나무림.
  - 아고산대 침엽수림 (1400–2100 m): 짙은 청록색의 구상나무류(*Abies veitchii*, Veitch fir).
  - 삼림한계선 경계부 (2100–2400 m): 일본잎갈나무(*Larix kaempferi*) 및 화산재 토양.

### 3. 수중 광학 및 에도 시대 대기 효과 (`shaders/ocean.frag`)
- **Jerlov Coastal II 광학 모델**: 도쿄만 하구의 혼합된 탁도와 일치하는 분광 감쇄 지수($\Sigma = (0.115, 0.032, 0.058)\text{ m}^{-1}$)를 적용했습니다.
- **호쿠사이 프러시안 블루**: 프러시안 블루(*Berliner Blau*) 팔레트와 산업화 이전의 해양 에어로졸에 맞추어 부하(subsurface) 산란 및 심해 용승색을 정밀 조정했습니다.
- **해무(Sea-Vapor Mist)**: 산업화 이전 우라가 수로의 높은 습도를 모사하는 대기 산란 효과를 적용했습니다.

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
