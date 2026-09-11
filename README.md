# AIPilot RL

2026-09-11 완료된 Head-on 런타임의 CUDA PPO / active league 소스입니다.
3-9와 Head-on을 명시적으로 선택할 수 있으며 원본 학습 디렉터리와 분리되어 있습니다.
GPU 리그전은 [evaluation/README.md](evaluation/README.md)를 참고하세요.

## 코드·모델 관리

[폴더 구조와 실행법](docs/STRUCTURE.md) · [커밋 규칙](docs/CONTRIBUTING.md)

새 학습은 `python scripts/train.py --scenario headon --run-name NAME --dry-run -- ...`
으로 경로와 인자를 먼저 확인합니다. `three_nine → artifacts/models/rl/3-9`,
`headon → headon`, `mixed → common`으로 저장을 분리합니다.
기존 절대경로 기반 resume 명령은 그대로 유지됩니다.

BT·MPC는 `artifacts/models/bt`, `mpc`, 팀원 패키지는 `external/inbox` 및
`external/approved`로 분리합니다. 보관만으로 실행되거나 리그전에 자동 추가되지 않습니다.
가중치와 외부 코드는 Git에 올리지 않습니다.

## 포함 범위

- `cuda_fdm/`: GPU 환경, 관측·보상, PPO, exploiter, payoff/Nash, active roster 및 historical recovery, 평가·중단 제어와 테스트
- `claude_code/`: 관측·보상 참조 구현, actor/critic 및 관련 코드
- `src/dogfight/`, `GeoMathUtil.py`: 환경 규약과 기하 계산 지원 코드

가중치, optimizer 상태, archive 정책 파일, W&B 로그·인증정보, 실행 로그 및 학습 데이터는 포함하지 않습니다. 이 저장소만으로 학습을 resume할 수는 없습니다. 별도 실행·재개 설정과 원본 학습 데이터가 필요합니다.

## 실행 환경

현재 CUDA 로더는 Windows NVIDIA 드라이버와 CUDA 지원 PyTorch에 포함된 NVRTC DLL을 사용합니다. Linux 호환성을 검증한 패키지는 아닙니다.

진입점: `python -m cuda_fdm.train_gpu --help`

CPU 회귀검증: `python -m cuda_fdm.tests.vnext_cpu_suite`

2026-09-11 정리 후 학습·pool 회귀 111개, 평가 5개, 저장 경로·Git 제외 규칙 6개로
총 122개 테스트가 통과했습니다. 새 학습은 시작하지 않았으며, 진행 중인 분리 리그전의
실행기·정책 어댑터·GPU 평가 함수·환경 파일 해시가 유지됨을 확인했습니다.

학습 실행에는 목적에 맞는 명시적 CLI 설정이 필요합니다. 기본값이 현재 캠페인 설정과 같다고 가정하지 마세요. W&B 사용 시 각자 자신의 계정으로 로그인해야 합니다.

기존 `cuda_fdm/README.md` 및 하위 설계 문서는 과거 실험 기록을 포함하므로 현재 코드/명시적 실행 설정을 우선합니다. 일부 과거 평가 도구는 별도로 보관된 모델, CPU 시뮬레이터 또는 데이터 파일을 요구합니다.

## 출처와 배포

기존 소스의 저작권·출처 표기는 보존합니다. F16/JSBSim 관련 구현을 포함하므로 이 스냅샷 전체에 임의의 오픈소스 라이선스를 새로 부여하지 않았습니다. 외부 공개/재배포 전 원본 및 파생 자료의 라이선스를 별도로 확인하세요.
