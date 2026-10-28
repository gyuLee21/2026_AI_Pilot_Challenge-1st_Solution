# AI Pilot — Autonomous Air Combat

**2026 AI Pilot Top Gun Challenge 1위 · 추락도락이다**

F-16 1대1 공중전을 위한 **CUDA 병렬 비행동역학, PPO·리그 자가대전, MPC·BT,
CPU 평가와 제출 실행파일**을 연결한 개발 포트폴리오입니다.
공통 기반·대회 제공 코드와 프로젝트에서 발전시킨 부분의 출처를 구분합니다.

## 핵심 개발 영역

| 영역 | 구현과 설계 | 코드 |
|---|---|---|
| 병렬 시뮬레이션 | F-16에 사용한 JSBSim 핵심 동역학·FCS의 CUDA 배치 실행 | [`cuda_fdm/gen`](cuda_fdm/gen), [`ref`](cuda_fdm/ref) |
| 강화학습 | 독립 MLP actor/critic, PPO·GAE, 4채널×21-bin 행동, 미래 위치 보조학습 | [`ppo_gpu.py`](cuda_fdm/ppo_gpu.py), [`future_aux.py`](cuda_fdm/future_aux.py) |
| 상대 관리 | Cold Archive와 bounded active pool, payoff/Nash, exploiter, 과거 상대 재검사 | [`league_vnext`](cuda_fdm/league_vnext), [최종 3-9 계보](research/competition_36500/README.md) |
| 제어기 | CEM MPC·reduced-order predictor, SDK 기반 커스텀 BT node | [`controllers/mpc`](controllers/mpc/README.md), [`controllers/bt`](controllers/bt/README.md) |
| 평가·배포 | 시나리오별 cross-play, 관측·주기·정규화 검증, CPU 추론·UDP·EXE 패키징 | [`evaluation`](evaluation/README.md), [`submission`](submission/README.md) |

## 먼저 읽을 문서

- [설치·실행·검증](docs/REPRODUCIBILITY.md)
- [논문과 실제 코드·checkpoint 대조](docs/PAPER_CODE_AUDIT.md)
- [코드와 모델 저장 구조](docs/STRUCTURE.md)
- [출처·권리·재현 범위](docs/PROVENANCE.md)
- [커밋 규칙](docs/CONTRIBUTING.md)

## 서로 다른 학습 계보를 섞지 않습니다

| 실행 경로 | 관측·용도 | 선택 |
|---|---|---|
| 기존 루트 `cuda_fdm/` | 214D 계열, 기존 Head-on/vNext 호환 경로 | `--runtime current` (기존 기본값) |
| `research/competition_36500/runtime/` | 최종 3-9 36.5k 계보의 **184D** 학습 소스 복원 | `--runtime competition36500` |

3-9 최종 actor와 critic은 모두 184차원입니다. 상대 행동 20차원을 critic에만
추가한 구조가 아닙니다. 보조 예측 head와 214D 가속도 확장은 별개입니다.
최종 계보의 연료·FCS·초기화·배치 payoff 변경과 실제 저장 설정은
[복원 기록](research/competition_36500/README.md)에 있습니다.

```powershell
# 저장소 루트에서 실행. 이 명령은 학습이나 파일 생성을 하지 않습니다.
python scripts/train.py --runtime competition36500 --scenario three_nine --run-name example184 --config configs/training/competition36500.example.json --dry-run

# 기존 214D 실행 경로도 유지됩니다. 값은 과거 캠페인 재현 설정이 아닙니다.
python scripts/train.py --scenario headon --run-name example214 --dry-run -- --iters 20000 --rollout 64
```

위 예시는 **새 실험의 명시적 설정**입니다. 새 run만으로 과거 제출 가중치나
모든 수동 상대 투입·보상 변경 이력을 재현했다고 주장하지 않습니다.
기존 checkpoint 재개에는 그 checkpoint의 archive·실행 상태와 검증된 설정이 필요합니다.

## 대회 DLL과 로컬 실행

대회 제공 DLL 3개를 [`third_party/competition_runtime`](third_party/competition_runtime/README.md)에
원본 바이트로 보관했습니다. 파일별 SHA-256, 비교한 공개 원본 commit 및
[권리 유의사항](third_party/competition_runtime/NOTICE.md)을 함께 기록합니다.
DLL은 이 프로젝트의 자체 구현이나 새 오픈소스 라이선스 대상으로 취급하지 않습니다.

```powershell
# 해시만 검증; DLL을 로드하지 않습니다.
python scripts/prepare_runtime.py

# 선택 사항: 별도 로컬 runtime에 복사. 다른 기존 파일은 덮어쓰지 않습니다.
python scripts/prepare_runtime.py --install
```

Python 3.11 / Windows를 기준으로 합니다. GPU 학습은 별도 NVIDIA 드라이버와
CUDA PyTorch·NVRTC가 필요하고, DLL에는 추가 CRT 의존성이 있을 수 있습니다.
제출 EXE는 simulator DLL을 호출하는 학습/CPU 평가 경로와 다릅니다.
모델과 제출 ZIP·서버 배포물은 [Releases](https://github.com/gyuLee21/2026_AI_Pilot_Challenge-1st_Solution/releases)와
로컬 원본을 구분해 관리합니다. 현재 저장소에 가중치나 서버 전체를 새로 올리지는 않습니다.

## 코드 구조

```text
cuda_fdm/                기존 214D GPU 환경 · PPO · league
research/competition_36500/
  runtime/               최종 184D 3-9 학습 소스와 해시 기록
claude_code/             기존 계보의 정책 · 관측 호환 구현
controllers/{mpc,bt}/    제어기 소스 · 설정 · native predictor
src/dogfight/            공통 상태/행동 계약 · UDP 인터페이스
evaluation/             평가 실행기 · 정책 어댑터 · 회귀 테스트
submission/             추론 · 통신 · EXE 빌드/검증
third_party/            출처를 구분한 대회 runtime
scripts/ configs/       명시적 실행기 · 설정 예시
tests/ docs/            검증 · 구조/방법론 문서
artifacts/              모델 · 팀원 파일 · 결과 등 로컬 전용 자료
```

기존 import 경로와 완료된 학습 폴더를 무리하게 이동하지 않습니다.
과거 캠페인 전용 스크립트는 연구 기록이며 재사용 가능한 일반 API와 구분합니다.
PT/PTH·로그·개인 경로·자격증명·빌드 부산물은 Git에서 제외하고,
DLL 예외는 위에서 검증한 정확한 3개 경로로만 제한합니다.

## 검증과 결과 해석

```powershell
python -m unittest discover -s tests -p "test_runtime_bundle.py" -v
python -m unittest discover -s tests -p "test_competition_runtime.py" -v
python -m unittest discover -s tests -p "test_managed_training.py" -v
python -m unittest discover -s tests -p "test_storage_policy.py" -v
```

이 검증은 GPU·가중치·DLL 로딩 없이 수행합니다. 전체 학습·교전 통합 검증은 별도입니다.
학습 pool 승률, CPU/GPU 결과, 10Hz/60Hz 결정 주기, rollout 길이가 다른 결과를
동일 성능 지표로 섞지 않습니다. FDM-only 처리량 배율도 전체 PPO 학습 가속률이 아닙니다.

커밋은 `fix`, `feat`, `refactor`, `test`, `docs`, `chore`로 변경 목적을 구분하며,
기반 코드·JSBSim 모델 데이터의 기존 출처와 권리 표시는 유지합니다.
