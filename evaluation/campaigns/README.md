# 과거 평가 캠페인 스크립트

대회 기간에 특정 체크포인트 · 경로 · 후보 조합으로 한 번 실행한 평가 스크립트입니다.
당시 결과를 다시 확인하거나 재현할 때만 쓰고, 새 실험은 상위 폴더의
`tournament.py`, `parallel_tournament.py`, `native_baselines.py`에서 시작하세요.

| 파일 | 용도 |
|---|---|
| `after_training.py`, `start_after_training.ps1` | 학습 정상 종료를 확인한 뒤 GPU 평가를 한 번 실행 |
| `transfer_headon46k.py`, `render_transfer_comparison.py` | 36k 전이 평가에 Head-on 46k 상대 추가, 결과 그림 |
| `cpu_36k_rate_match.py` | 36k 모델의 10Hz / 60Hz CPU 판단 주기 비교 |
| `continue_power_tests.py`, `power_results.py` | CPU 대전 큐 실행과 결과 집계 |
| `diagnose_taemin.py` | 팀원 모델 연결 진단 |

스크립트는 저장소 루트에서 실행합니다(예: `python evaluation/campaigns/power_results.py`).
대부분 로컬 `artifacts/` 아래의 당시 모델 · 결과 폴더를 전제로 합니다.
