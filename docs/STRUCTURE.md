# 코드 · 모델 저장 구조

## 코드

```text
cuda_fdm/                     GPU 환경, PPO, 리그, 학습 CLI (214D 계보)
research/competition_36500/   최종 3-9 학습 소스와 설정·해시 기록 (184D 계보)
claude_code/                  체크포인트 호환 네트워크, 관측·보상 정의
controllers/mpc/              MPC 소스 · 설정 · native 예측기 · 테스트
controllers/bt/               커스텀 BT 노드 (원본 SDK 필요)
src/dogfight/                 공통 환경 규약, 기하 계산, UDP 인터페이스
evaluation/                   리그전 실행기, 정책 어댑터, 테스트
submission/                   추론 · 통신 규약, 패키징
scripts/                      학습 · 평가 · 런타임 준비 실행기
configs/                      학습 · 평가 · 모델 등록 예시
third_party/                  대회 제공 런타임 일부 (출처 분리)
tests/                        저장 구조 · 실행기 테스트
artifacts/                    로컬 전용 (README만 Git 관리)
```

`evaluation/` 안의 `after_training.py`, `transfer_*.py`, `final_selection.py` 등은
특정 대회 기간의 평가 기록용 스크립트입니다. 새 실험은 `tournament.py` · `native_baselines.py`에서 시작합니다.

## 로컬 모델 저장 (`artifacts/`, Git 제외)

```text
artifacts/models/
  rl/3-9/<run-name>/          3-9 학습 결과
  rl/headon/<run-name>/       Head-on 학습 결과
  rl/common/<name>/           공통 기준 모델
  bt/<name>/                  BT DLL + XML
  mpc/<name>/                 MPC 스냅샷
  external/inbox|approved/    팀원 모델 (검토 전 / 검토 후)
artifacts/evaluations/        평가 결과
artifacts/runtimes/           격리 설치된 대회 런타임
```

`scripts/train.py`로 시작한 학습은 `artifacts/models/rl/<시나리오>/<run-name>/`에
`checkpoint.pt`, `iter_*.pt`, `training.csv`, `league/`를 저장합니다.
비어 있지 않은 run 폴더는 `--resume` 없이 덮어쓰지 않습니다.

## 외부 모델

팀원 모델은 코드와 가중치를 함께 `external/inbox`에 두고, 출처 · 해시 · 관측/행동 정의를 기록한 뒤
검토를 거쳐 `approved`로 옮깁니다. 신뢰할 수 없는 PyTorch/pickle 파일은 열람 목적이라도 로드하지 않습니다.
