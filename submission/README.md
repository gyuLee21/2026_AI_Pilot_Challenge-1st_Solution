# Submission packaging and wire contracts

이 폴더는 추론·UDP·패키징 소스이며 제출 ZIP 보관소가 아닙니다.

- `frozen_policy.py`: actor-only CPU 추론
- `decision_rate.py`: 상태 응답 주기와 정책 의사결정 주기 분리
- `main_client.py`: 36.5k 메인용 10Hz 판단 / 60Hz 응답 진입점
- `headon_client.py`: **과거 46k 60Hz 진입점** 및 공통 패킷 매핑
- `build_headon.py`: CPU PyTorch 환경에서 main/headon one-file EXE 빌드
- `verify_*`: 로컬 wire 및 실행파일 검증 도구

주의: 최종 46.5k 10Hz Head-on ZIP은 별도 격리 빌드에서 제작됐습니다.
이 폴더의 과거 Head-on 진입점을 그대로 실행해 최종 제출물을 재생성하면 안 됩니다.
릴리스마다 payload SHA256, iteration, 판단 주기, 팀명, wire 검증 결과를 함께 확인해야 합니다.
이번 저장소 정리는 기존 제출 ZIP이나 가중치를 수정하지 않습니다.

10Hz 판단 / 60Hz 응답은 6개 상태 프레임마다 새 제어를 계산하고 그 사이에는
이전 명령을 유지하는 방식입니다. 응답 횟수와 신경망 추론 횟수는 다릅니다.
실행파일 생성은 별도 CPU-only PyTorch/PyInstaller 환경에서 하며,
학습 환경의 CUDA 라이브러리를 그대로 묶지 않습니다.

`tests/test_decision_rate.py`는 판단 주기 계약을 검사합니다.
`tests/test_main_submission.py`는 로컬 정책 가중치를 사용하는 통합 테스트입니다.
저장소만 clone한 상태에서는 해당 가중치 테스트를 실행할 수 없습니다.
