# AIPilot RL

현재 3-9 캠페인에서 사용하는 CUDA PPO / active league의 소스 스냅샷입니다.
원본 학습 디렉터리와 분리해 복사했으며 실행 중인 학습에는 변경을 가하지 않았습니다.

## 포함 범위

- `cuda_fdm/`: GPU 환경, 관측·보상, PPO, exploiter, payoff/Nash, active roster 및 historical recovery, 평가·중단 제어와 테스트
- `claude_code/`: 관측·보상 참조 구현, actor/critic 및 관련 코드
- `src/dogfight/`, `GeoMathUtil.py`: 환경 규약과 기하 계산 지원 코드

가중치, optimizer 상태, archive 정책 파일, W&B 로그·인증정보, 실행 로그 및 학습 데이터는 포함하지 않습니다. 이 저장소만으로 현재 학습을 resume할 수는 없습니다. 준비 중인 별도 head-on 런타임/자동 실행기는 이번 스냅샷에 포함하지 않았습니다.

## 실행 환경

현재 CUDA 로더는 Windows NVIDIA 드라이버와 CUDA 지원 PyTorch에 포함된 NVRTC DLL을 사용합니다. Linux 호환성을 검증한 패키지는 아닙니다.

진입점: `python -m cuda_fdm.train_gpu --help`

CPU 회귀검증: `python -m cuda_fdm.tests.vnext_cpu_suite`

2026-09-08 복사본에서 CLI import 및 CPU 테스트 109개 통과를 확인했습니다. GPU 학습은 기존 실행과 자원 충돌을 피하기 위해 별도로 시작하지 않았습니다.

학습 실행에는 목적에 맞는 명시적 CLI 설정이 필요합니다. 기본값이 현재 캠페인 설정과 같다고 가정하지 마세요. W&B 사용 시 각자 자신의 계정으로 로그인해야 합니다.

기존 `cuda_fdm/README.md` 및 하위 설계 문서는 과거 실험 기록을 포함하므로 현재 코드/명시적 실행 설정을 우선합니다. 일부 과거 평가 도구는 별도로 보관된 모델, CPU 시뮬레이터 또는 데이터 파일을 요구합니다.

## 출처와 배포

기존 소스의 저작권·출처 표기는 보존합니다. F16/JSBSim 관련 구현을 포함하므로 이 스냅샷 전체에 임의의 오픈소스 라이선스를 새로 부여하지 않았습니다. 외부 공개/재배포 전 원본 및 파생 자료의 라이선스를 별도로 확인하세요.
