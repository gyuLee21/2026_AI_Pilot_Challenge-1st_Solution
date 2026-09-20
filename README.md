# 2026 AI Pilot Challenge — 1st Solution

2026 AI Pilot Top Gun Challenge에서 **추락도락이다** 팀으로 1위를 기록한 자율 공중전 에이전트 개발 기록입니다.
이 저장소는 F-16 1대1 dogfight 환경에서 **강화학습(PPO·active league), 모델 예측 제어(MPC), 행동 트리(BT)**를 학습·평가·제출 파이프라인으로 연결한 공개 기술 저장소입니다.

> [!NOTE]
> 이 저장소는 같은 팀의 1위 솔루션을 구성한 공개 가능한 코드와 설계·검증 문서를 중심으로 정리한 포트폴리오/연구 기록입니다. 대회 SDK·DLL, 팀원 모델, 실행 중인 학습 로그와 대용량 산출물은 소스 저장소가 아니라 GitHub Releases 또는 로컬 원본으로 관리합니다.

## 프로젝트 개요

문제는 제한된 관측과 제어 입력으로 상대 전투기의 위치·에너지·공격 기하를 유리하게 만들면서 생존과 명중을 달성하는 것입니다. 단일 정책 네트워크만 학습하는 것으로는 실제 제출까지 이어지지 않으므로 다음 계층을 함께 관리했습니다.

1. JSBSim 기반 F-16 동역학과 dogfight 환경
2. GPU 병렬 물리·PPO 학습과 active league self-play
3. MPC와 BT 기반의 별도 제어 정책 계열
4. CPU/GPU·10Hz/60Hz·시나리오별 평가 도구
5. 관측·행동·UDP 계약을 지키는 제출 bundle·실행파일 검증

## 주요 구성

| 영역 | 역할 | 시작점 |
|---|---|---|
| RL | CUDA 병렬 환경, PPO, MLP/GRU, 관측·보상 | [`cuda_fdm`](cuda_fdm/README.md) |
| League | active pool, exploiter, payoff/Nash, 과거 상대 재검사 | [`cuda_fdm/league_vnext`](cuda_fdm/league_vnext) |
| MPC | CEM 탐색, 표적 운동 예측, reduced-order F-16 predictor | [`controllers/mpc`](controllers/mpc/README.md) |
| BT | 기하·위협 판단, 기동 선택, blackboard 커스텀 노드 | [`controllers/bt`](controllers/bt/README.md) |
| Evaluation | cross-play, CPU/GPU 비교, 결정 주기 비교, 회귀검증 | [`evaluation`](evaluation/README.md) |
| Submission | CPU 추론, UDP, 패키징, 실행 계약 검증 | [`submission`](submission/README.md) |
| Shared contracts | 상태·행동 schema, provider, UDP/Unreal 인터페이스 | [`src/dogfight`](src/dogfight) |

## 학습과 제어 파이프라인

### PPO와 active league

`cuda_fdm`은 여러 환경을 GPU에서 병렬 실행하고 PPO rollout/update를 수행합니다. 상대 정책 하나에만 맞춰지는 것을 막기 위해 최신 정책, 최근 정책, 핵심 과거 정책, challenger와 같은 역할을 가진 상대 pool을 운영합니다.

정책 선택은 학습 pool 승률 하나로 결정하지 않습니다. 학습에 사용하지 않은 상대, 강한 과거 상대, 서로 다른 전략을 포함한 cross-play에서 다음 지표를 분리해 기록합니다.

- win/draw/loss
- 상대별 성적과 전략군별 성적
- crash rate와 ground-loss rate
- HP 차이, timeout, 종료 원인
- 물리·관측 계약 회귀 결과와 CPU 추론 지연

iteration이 더 크거나 checkpoint가 최신이라는 이유만으로 final best라고 부르지 않습니다. learner snapshot, stable champion, challenger의 역할과 provenance를 함께 기록합니다.

### 관측·행동 계약

학습과 제출 경로는 같은 모델을 사용하더라도 관측 생성 시점, throttle 변환, action history, normalization이 조금만 달라지면 다른 입력을 만들 수 있습니다. 이 저장소에서는 시나리오와 관측 계열을 명시적으로 분리합니다.

| 항목 | 의미 |
|---|---|
| `three_nine` | 두 기체가 옆에서 서로 반대 방향을 보는 초기 기하 |
| `headon` | 두 기체가 정면으로 마주 보는 초기 기하 |
| `184D` | 기존 `claude164r` 관측 계열과 history를 사용하는 계약 |
| `214D` | 가속도 관련 feature를 확장한 장기 학습 런타임 계약 |
| `10Hz` | 정책이 새 제어 결정을 내리는 주기 |
| `60Hz` | 물리 시뮬레이션과 제어 응답을 진행하는 주기 |

체크포인트와 bundle에는 가능한 한 시나리오·관측 차원·정책 결정 주기·변환 규칙을 함께 남겨 서로 다른 계약을 조용히 섞지 않도록 합니다.

### MPC

`controllers/mpc/release`는 공중전 기하와 비행 envelope를 비용함수에 넣는 CEM 기반 MPC입니다.

- 2초 예측 지평
- 10Hz 재계획, 60Hz 제어 출력
- 0.5초 간격 4개 action knot: roll, pitch, rudder, throttle
- 후보 48개, 2회 반복, 상위 1/6 elite 갱신
- symmetric noise와 maneuver library
- target motion prediction, 가속도·선회율 제한, smoothing
- damage, attack geometry, closure, nose advantage, threat, range, overshoot, 지면·envelope risk, slew 비용

native predictor는 reduced-order F-16 동역학, 공력 테이블/XML, F100 thrust, ISA atmosphere, quaternion 적분, FCS PID·actuator·engine spool을 사용합니다. 좌표계는 NED와 body-frame 속도를 일관되게 유지하고, relative-rate는 SO(3) 관점에서 처리합니다.

### 행동 트리(BT)

`controllers/bt`는 공격·방어·이탈·위협 대응과 같은 전술 상태를 명시적으로 구성하는 BT 계열입니다. custom node와 blackboard를 원본 AIP DCS SDK overlay로 적용하며, 독립 실행 프로그램이 아니라 해당 SDK·rule XML·DLL 계약과 함께 사용합니다.

## 평가 설계

평가 결과는 다음 축을 분리해야 해석할 수 있습니다.

1. **물리 회귀** — 동일 상태·입력에서 CPU 기준 구현과 CUDA 구현의 trajectory를 비교합니다.
2. **정책 비교** — 동일 초기조건, 역할 교환, 동일 상대 pool에서 cross-play합니다.
3. **시나리오 비교** — `three_nine`과 `headon`을 섞어 평균내지 않고 별도로 봅니다.
4. **실행 경로 비교** — CPU/GPU physics와 10Hz/60Hz decision rate의 차이를 분리합니다.
5. **제출 검증** — 최종 bundle을 CPU-only provider와 UDP 실행 경로로 확인합니다.

학습 중간 로그, 특정 seed, 제한된 상대 pool의 승률은 전체 대회 성능과 동일하지 않습니다. 각 실험의 입력, opponent roster, 게임 수, 종료 조건, 정책 결정 주기를 함께 확인해야 합니다.

## 설치와 빠른 실행

Windows 개발 환경을 기준으로 합니다. GPU 학습에는 NVIDIA 드라이버, CUDA 지원 PyTorch, NVRTC가 필요하며, CPU native 평가와 실제 제출에는 대회 SDK와 native runtime이 별도로 필요합니다.

모든 명령은 저장소 루트에서 실행합니다. `aircraft/`, `engine/`, rule XML과 native DLL이 상대경로를 사용하므로 다른 작업 디렉터리에서 실행하면 실패할 수 있습니다.

```powershell
# 실제 학습은 시작하지 않고 인자·경로만 검증
python scripts/train.py --scenario three_nine --run-name experiment01 --dry-run -- --iters 20000 --rollout 64

# 보유한 모델 경로를 evaluation/models.template.json에 넣은 뒤 입력 검증
python scripts/evaluate.py --spec evaluation/models.template.json --output artifacts/evaluations/example --validate-only

# 기본 회귀검증
python -m unittest discover -s tests
python -m unittest discover -s evaluation -p "test_*.py"
python -m cuda_fdm.tests.vnext_cpu_suite
```

### 저장 경로

새 학습 산출물은 시나리오별로 다음 로컬 경로를 사용합니다.

```text
artifacts/models/rl/3-9/
artifacts/models/rl/headon/
artifacts/models/rl/common/
```

기존 학습을 재개하려면 원본 checkpoint, archive, 실행 설정과 그 checkpoint가 속한 관측 계약이 필요합니다. CLI 기본값을 과거 캠페인의 재현 가능한 설정으로 간주하지 않습니다.

## 최종 제출파일과 로컬 교전서버

대용량 ZIP은 Git의 일반 파일로 관리하지 않고 GitHub Releases에 함께 게시합니다. 현재 공개할 release asset은 다음 세 개입니다.

| 파일 | 용도 |
|---|---|
| `추락도락이다_APTGC2026_main.zip` | 3-9/main 시나리오 최종 제출 패키지 |
| `추락도락이다_APTGC2026_headon.zip` | head-on 시나리오 최종 제출 패키지 |
| `BattleServer_V1.2_VeryLow.zip` | 로컬에서 제출 agent를 실행·관찰하기 위한 교전 서버 |

Release asset을 내려받은 뒤 압축을 풀어 로컬 서버 폴더와 제출 패키지를 각각 준비합니다. 서버와 제출파일은 Windows native runtime, DLL, 모델 bundle과 강하게 결합되어 있으므로 저장소 루트에 무작정 압축을 풀기보다 release 설명의 실행 순서를 따릅니다.

최종 제출 bundle을 새로 만들 때는 다음과 같이 실행합니다.

```powershell
# CPU snapshot → submission bundle
python claude_code/snapshot_to_bundle.py --snapshot-dir <snapshot-dir> --output-dir artifacts/cpu_ppo_final

# CUDA checkpoint → submission bundle
python -m cuda_fdm.gpu_ckpt_to_bundle --ckpt <checkpoint.pt> --output-dir artifacts/gpu_ppo_final
```

실제 대회 제출파일은 이미 고정된 release artifact이므로, 재빌드 결과를 원본 제출파일과 동일하다고 간주하지 않습니다. 모델·payload hash·팀명·시나리오·wire contract를 확인한 뒤 별도 파일로 취급합니다.

## 모델 bundle과 제출 계약

제출 단계에서는 다음을 확인합니다.

- observation/action dimension과 normalization 계약
- throttle과 action history의 의미
- 10Hz 정책 결정과 60Hz 응답 유지
- CPU-only latency
- UDP wire format, reset, timeout, 종료 조건
- bundle metadata, checkpoint provenance, payload hash
- 패키징된 실행파일이 저장소 루트를 기준으로 필요한 asset을 찾는지

모델 가중치와 최종 제출 ZIP의 원본은 Git history에 넣지 않고 release asset으로 관리합니다. 대회 SDK와 native server는 재배포 권한과 공개 범위를 확인한 뒤 release에 게시합니다.

## 저장소 구조

```text
cuda_fdm/           GPU 물리 · PPO · active league
claude_code/        checkpoint 호환 모델 · 관측 참조 구현
controllers/
  mpc/release/      MPC 설정 · native predictor · 테스트
  bt/aip_dcs/       SDK에 적용하는 BT overlay
src/dogfight/       상태·행동 계약 · UDP/Unreal 인터페이스
evaluation/         평가 실행기 · policy adapter · 회귀검증
submission/         추론 · 통신 · 패키징 · 실행파일 검증
scripts/            학습·평가 진입점과 실험 도구
configs/            재사용 가능한 설정 예시
tests/              저장 경로·추론 계약 테스트
docs/               구조·출처·기여·정리 검증 기록
artifacts/          로컬 모델·SDK·결과 저장 경로 (대용량 산출물 제외)
```

## 문서 안내

- [`docs/STRUCTURE.md`](docs/STRUCTURE.md) — 공개 저장소의 구조와 로컬 전용 산출물
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — 코드·물리 모델·재현 범위와 출처
- [`docs/PORTFOLIO_CLEANUP.md`](docs/PORTFOLIO_CLEANUP.md) — 공개 전 정리 원칙과 검증 기록
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — 변경·커밋 규칙
- [`cuda_fdm/ARCHITECTURE_SEARCH.md`](cuda_fdm/ARCHITECTURE_SEARCH.md) — PPO 구조 탐색 설계
- [`cuda_fdm/MLP_SIZE_SEARCH.md`](cuda_fdm/MLP_SIZE_SEARCH.md) — 네트워크 크기 비교 기준
- [`evaluation/README.md`](evaluation/README.md) — 평가 실행과 결과 해석
- [`submission/README.md`](submission/README.md) — CPU 추론과 패키징 계약

## 공개 범위와 한계

이 공개 저장소는 다음을 목표로 합니다.

- 학습·제어·평가 구조를 읽고 검토할 수 있게 하기
- 실험 결과를 상대 pool, seed, 시나리오, 결정 주기와 함께 해석하게 하기
- 대회 SDK나 대용량 산출물 없이도 소스 구조와 계약을 이해하게 하기

다음은 Git history에 넣지 않습니다.

- 개인·팀 모델 가중치와 checkpoint
- 대회 SDK, native DLL, 실행파일, 라이선스가 제한된 simulator asset
- W&B·실험 로그·대용량 dataset
- 실제 서버 자격증명, token, 로컬 절대경로

최종 제출 ZIP과 로컬 교전서버처럼 공개가 허용된 대용량 artifact는 GitHub Releases에서 별도로 관리합니다. 따라서 `python` 명령이 모든 환경에서 즉시 완주한다고 보장하지 않으며, 실제 재현에는 Windows, CUDA/PyTorch, native simulator, 대회 SDK와 release asset이 필요할 수 있습니다.

## 개발·출처

기존 기반 코드와 F-16/JSBSim 관련 자료의 출처와 라이선스 범위는 [`docs/PROVENANCE.md`](docs/PROVENANCE.md)에 기록합니다. 저장소 전체에 새로운 오픈소스 라이선스를 임의로 적용하지 않으며, 외부 공개·재배포가 필요한 항목은 별도로 권한을 확인해야 합니다.

커밋은 다음 접두사를 사용합니다.

```text
feat:     기능 추가
fix:      오류 수정
refactor: 구조 변경
test:     검증 추가·수정
docs:     문서 변경
chore:    관리 작업
```
