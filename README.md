# AI Pilot — Learning, Control & Evaluation

AI Pilot Top Gun Challenge 공중전 에이전트 개발 기록입니다.
**강화학습(PPO·리그 학습), 모델 예측 제어(MPC), 행동 트리(BT)**를 다루며,
학습부터 상대별 평가와 제출 실행파일 패키징까지의 코드를 관리합니다.

이 저장소는 **비공개 소스 저장소**입니다. 모델 가중치, 대회 SDK/DLL, 팀원 모델,
실험 로그와 최종 제출 ZIP은 Git에 포함하지 않습니다.

## 주요 구성

| 영역 | 구현 내용 | 시작점 |
|---|---|---|
| RL | CUDA 병렬 환경, PPO, MLP/GRU, 관측·보상 | [cuda_fdm](cuda_fdm/README.md) |
| League | exploiter, payoff/Nash, 제한된 active pool, 과거 상대 재검사 | [league_vnext](cuda_fdm/league_vnext) |
| Evaluation | 3-9/Head-on 분리, 정책 어댑터, CPU/GPU 평가, 판단 주기 비교 | [평가 가이드](evaluation/README.md) |
| MPC | CEM 탐색, 표적 예측, C++ reduced-order predictor | [Release MPC](controllers/mpc/README.md) |
| BT | 기하·위협 판단, 기동 선택, blackboard 커스텀 노드 | [BT 소스](controllers/bt/README.md) |
| Submission | CPU 추론, 판단/응답 주기 분리, UDP, 단일 EXE 패키징 | [패키징 가이드](submission/README.md) |

## 폴더 구조

```text
cuda_fdm/           GPU 환경 · PPO · 리그 학습 (기존 import 경로 유지)
claude_code/        체크포인트 호환 모델 · 관측 참조 구현
controllers/
  mpc/release/      MPC 소스 · 설정 · native predictor · 테스트
  bt/aip_dcs/       대회 SDK에 적용하는 커스텀 BT 소스 overlay
src/dogfight/       공통 상태/제어 계약 · UDP 인터페이스
evaluation/        평가 실행기 · 정책 어댑터 · 회귀 테스트
submission/        추론/통신 · 패키징 · 실행파일 검증
scripts/           학습/평가 진입점 및 실험 분기 도구
configs/           재사용 가능한 설정 예시
tests/             저장 경로 · 추론 계약 테스트
docs/              구조 · 개발 규칙 · 실험/검토 기록
artifacts/          로컬 전용 모델 · SDK · 결과 (Git 제외)
```

## 실행

Windows 환경을 기준으로 개발했습니다. GPU 학습에는 NVIDIA 드라이버,
CUDA 지원 PyTorch 및 NVRTC가 필요합니다. CPU native 평가는 대회 SDK가 별도로 필요합니다.
단일 requirements 파일로 모든 대회 런타임이 설치되는 프로젝트는 아닙니다.

```powershell
# 인자와 저장 경로만 확인; 학습은 시작하지 않음
python scripts/train.py --scenario three_nine --run-name experiment01 --dry-run -- --iters 20000 --rollout 64

# 평가 명단에 보유한 모델 경로를 기입한 뒤 입력 검증
python scripts/evaluate.py --spec evaluation/models.template.json --output artifacts/evaluations/example --validate-only

# 기본 회귀검증
python -m unittest discover -s tests
python -m unittest discover -s evaluation -p "test_*.py"
python -m cuda_fdm.tests.vnext_cpu_suite
```

`three_nine`, `headon`, `mixed` 학습의 새 저장 경로는 각각
`artifacts/models/rl/3-9`, `headon`, `common`입니다.
기존 학습을 재개하려면 원본 checkpoint·archive와 해당 실행 설정이 필요합니다.
CLI 기본값을 과거 캠페인 설정으로 간주하지 마세요.

## 평가를 해석하는 기준

- 학습 pool 승률만으로 최종 모델을 선택하지 않습니다. 학습에 사용하지 않은 상대,
  과거 강한 상대, 서로 다른 전략을 분리해서 비교합니다.
- 승/패/무, 상대별 성적, 추락률, HP 차이를 함께 봅니다. 상대가 다른 평균 순위는 직접 비교하지 않습니다.
- CPU/GPU 물리 및 초기화 차이와 10Hz/60Hz 의사결정 차이를 별도로 확인합니다.
  한 시나리오에서 좋은 판단 주기가 다른 모델에서도 좋다고 가정하지 않습니다.
- 과거 결과와 설계 문서는 연구 기록이며 현재 실행 설정이나 보편적 성능 보장이 아닙니다.

## 개발·출처

[상세 구조](docs/STRUCTURE.md) · [커밋 규칙](docs/CONTRIBUTING.md) ·
[소스 출처와 재현 범위](docs/PROVENANCE.md) · [정리 검증 기록](docs/PORTFOLIO_CLEANUP.md)

커밋은 `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`로 목적을 구분합니다.
기존 기반 코드와 F-16/JSBSim 관련 자료의 출처를 보존하며, 저장소 전체에 새로운
오픈소스 라이선스를 임의 적용하지 않습니다. 외부 공개 전 재배포 권한을 별도로 검토해야 합니다.
