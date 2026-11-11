# cuda_fdm — JSBSim F-16의 CUDA 포팅과 GPU 학습 루프

대회 환경의 CPU JSBSim v1.0.0 F-16 비행물리를 GPU로 옮겨, 수천 개의 1:1 교전을
동시에 시뮬레이션하고 PPO 데이터 수집과 갱신까지 GPU 안에서 끝내기 위한 모듈입니다.

> 이 폴더는 **214D 관측 계보**(Head-on 등)의 런타임입니다.
> 최종 3-9 36.5k(184D) 학습 소스는 [research/competition_36500](../research/competition_36500/README.md)에 따로 있습니다.

## 접근 방법

1. **Python 레퍼런스 (`ref/`)** — JSBSim F-16 단일기 데이터플로 전체를 double 정밀도 Python으로 평탄화
   (대기 · FCS 10채널 · 질량/관성 · F100 엔진 · 공력 40개 계수 · ECI/ECEF/Body 좌표 · WGS84 J2 중력 · 적분기).
2. **CUDA-C 생성 (`gen/`)** — 검증된 레퍼런스를 host/device 공용 C(`fdm.cuh`, 자동생성 `f16_gen.cuh`)로 이식하고
   NVRTC로 런타임 컴파일. CUDA 툴킷 없이 PyTorch 번들 NVRTC만으로 동작합니다.
3. **배치 RL 환경 (`gpu_env.py`, `rl_env.py`, `obs_reward.py`)** — 1 env = 2기, 대회 초기조건 분포,
   종료 판정과 GPU 내부 자동 리셋(CPU-GPU 동기화 없음).
4. **커널 융합 (`gen/obs_kernel.cu`)** — 관측 생성 · HP/연료 적분 · 보상 · 종료를 NVRTC 커널 2개로 융합.
5. **GPU 네이티브 PPO (`ppo_gpu.py`, `train_gpu.py`)** — rollout · GAE · 갱신 · 상대 추론을 모두 GPU 텐서로 처리,
   리그 상대 풀과 exploiter 학습 포함.

## 검증 정확도

6초 4채널 기동을 CPU JSBSim v1.0.0 매 스텝 기록(golden)과 비교했습니다.

| 비교 | 위치 | 자세 | 속도 | 받음각 |
|---|---|---|---|---|
| Python 레퍼런스 vs golden | 0.036 ft | 0.0030° | 0.017 fps | 0.0005° |
| GPU(NVRTC) vs golden | 0.0356 ft | 0.0030° | 0.017 fps | 0.00055° |
| GPU vs host-C | 0 ft | 2.9e-15 rad | 1.1e-12 fps | 5.5e-16 rad |

오차는 모두 마지막 스텝에 몰려 있는 double 누적 드리프트입니다. CPU/GPU 간 bit-exact는 FMA · libm 차이로
불가능하므로 tolerance 기반 회귀 테스트(`tests/regression_test.py`)로 고정했습니다.
GPU 관측 · 보상은 CPU 참조 구현과 1e-13 이하 차이로 일치합니다(`tests/obs_reward_val.py`).

## 성능 (RTX 3070 Ti)

| 항목 | 결과 |
|---|---|
| 물리 커널 단독 (FP64, 262k env) | 4,800만 env-step/s |
| 레지스터 상한 조정 (255 → 96) | occupancy 17% → 42%, 처리량 +24% |
| FP32 변환 옵션 (65k 기체, action repeat 4) | FP64 대비 7.3배 (위치만 6초에 43 ft 드리프트) |
| 전체 RL step (4,096 env, 관측 · 보상 · 자동 리셋 포함) | 3.68 ms — PyTorch 연산 버전(~50 ms) 대비 13.6배 |

## 포팅 시 주의점

- FCS 컴포넌트 dt는 적분 dt(1/60)의 절반인 1/120로 고정 캡처됩니다(원본 v1.0.0 동작).
- `FGMatrix33` 내부 배열은 column-major입니다.
- 중력은 WGS84 J2, `altitudeASL = |eci| − a` (측지고도 아님).
- 적분기: 사원수 · 각속도 = Rectangular Euler, 위치 = AB3, 속도 = AB2.
- `throttle_pos = 2 × throttle_cmd` (0.8 → military 1.0 + afterburner 0.6).
- 모델 순서: Propagate → Inertial → Atmosphere → FCS → MassBalance → Auxiliary → Propulsion → Aero → Accelerations.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `ref/` | 검증된 Python 레퍼런스 (`jsb_fdm.py` 통합 스텝) |
| `gen/` | CUDA-C 생성기 · 물리/관측 커널 |
| `gpu_env.py`, `rl_env.py`, `ic.py` | 배치 환경, 대회 초기조건, ECI↔NED 변환 |
| `obs_reward.py`, `reward_modes.py` | 관측 · 보상 배치 구현 |
| `ppo_gpu.py`, `train_gpu.py`, `future_aux.py` | PPO, 학습 CLI, 미래 위치 보조 학습 |
| `league.py`, `league_vnext/`, `exploiter_init.py` | 상대 풀 · Nash 선발 · exploiter |
| `gpu_ckpt_to_bundle.py` | 학습 체크포인트 → 제출용 bundle 변환 |
| `tests/` | golden 대조, 정합성, 회귀 테스트 |

## 실행

저장소 루트에서 실행합니다.

```powershell
# 회귀 테스트 (GPU 필요)
python cuda_fdm/tests/regression_test.py

# 학습 / 이어서 학습
python -m cuda_fdm.train_gpu --save runs/gpu.pt --log runs/gpu.csv
python -m cuda_fdm.train_gpu --resume runs/gpu.pt --save runs/gpu.pt --log runs/gpu.csv

# 체크포인트를 제출용 bundle로 변환
python -m cuda_fdm.gpu_ckpt_to_bundle --ckpt runs/gpu.pt --output-dir artifacts/models/my_bundle
```

관리형 실행 경로는 루트의 `scripts/train.py`를 권장합니다. 보조 학습 설계는
[FUTURE_AUXILIARY.md](FUTURE_AUXILIARY.md), 구조 탐색 기록은 [MLP_SIZE_SEARCH.md](MLP_SIZE_SEARCH.md)에 있습니다.
