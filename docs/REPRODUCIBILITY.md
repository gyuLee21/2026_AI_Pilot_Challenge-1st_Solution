# 설치 · 실행 가이드

## 환경 단계

| 단계 | 필요한 것 | 할 수 있는 것 |
|---|---|---|
| 1. 소스 검사 | Python 3.11, Git | 해시 · 실행기 · 저장 구조 테스트 (GPU · 가중치 · DLL 로드 없음) |
| 2. CPU 정책 · 리그 테스트 | `requirements.txt` | 리그 상태 로직, 정책 추론 |
| 3. CUDA 학습 | NVIDIA 드라이버, CUDA PyTorch(NVRTC 포함) | GPU 환경 · PPO 학습 |
| 4. CPU 교전 평가 | Windows x64, 대회 DLL · CRT, 모델 파일 | 대회 시뮬레이터 기반 평가 |
| 5. MPC · 제출 빌드 | Visual Studio 2022, CMake, PyInstaller | [MPC](../controllers/mpc/README.md) · [제출](../submission/README.md) 문서 참고 |

`requirements.txt`는 의존성 범위입니다. PyTorch는 시스템에 맞는 CPU/CUDA 빌드를 따로 고르세요.

## 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/prepare_runtime.py            # 대회 DLL 해시 검증만 수행
python scripts/prepare_runtime.py --install  # artifacts/runtimes/competition 에 격리 설치
```

설치 명령은 기존 파일과 내용이 다르면 덮어쓰지 않습니다. DLL 실행이나 CRT 설치는 하지 않습니다.

## 학습 계보 선택

| 계보 | 관측 | 선택 방법 |
|---|---|---|
| 루트 `cuda_fdm/` | 214D (Head-on 등) | 기본값 |
| `research/competition_36500/` | 184D (최종 3-9) | `--runtime competition36500` |

두 계보는 같은 이름의 모듈을 쓰지만 별도 인터프리터 · 작업 경로로 실행되어 한 프로세스에서 섞이지 않습니다.
항상 `--dry-run`으로 명령을 확인한 뒤 실행하세요.

```powershell
python scripts/train.py --runtime competition36500 --scenario three_nine --run-name my_run `
  --config configs/training/competition36500.example.json --dry-run
```

예시 설정은 새 실험을 위한 것이며, 제출 가중치를 그대로 재현하지는 않습니다.
과거 학습을 정확히 이어가려면 원본 체크포인트와 상대 archive · 실행 상태가 모두 필요합니다.

## 평가

```powershell
python scripts/evaluate.py --spec configs/evaluation/three_nine.local.json --output artifacts/evaluations/three_nine --validate-only
```

모델 경로는 로컬 평가 spec에 지정합니다([예시](../configs/evaluation/README.md)).
CPU/GPU 평가, 10Hz/60Hz 판단 주기는 서로 다른 조건이므로 결과를 따로 해석합니다.
자세한 프로토콜은 [evaluation/README.md](../evaluation/README.md)를 참고하세요.

## 테스트

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s evaluation -p "test_*.py"
```

일부 통합 테스트는 로컬 가중치나 빌드 도구가 있어야 실행됩니다.
