# Evaluation — GPU 리그전 · CPU 검증

## 재사용 실행기와 과거 캠페인

새 평가는 `tournament.py` / `parallel_tournament.py` 또는
`native_baselines.py` / `native_matchups.py`에서 시작하세요.
`decision_rate_match.py`는 CPU 판단 주기 비교를 담당합니다.
CPU native 경로에는 Git에 포함되지 않는 대회 SDK/DLL이 필요합니다.

`after_training.py`, `transfer_campaign.py`, `transfer_headon46k.py`,
`cpu_36k_rate_match.py`, `final_selection.py` 및 `start_after_*.ps1`은
특정 과거 캠페인 경로·iteration·후보를 담은 실행 기록입니다.
재현을 위해 유지하지만 새 실험에 그대로 실행하지 마세요.
결과는 항상 새로운 출력 경로에 저장하고, 기존 manifest와 다른 설정을 섞지 않습니다.

## GPU 리그전

학습과 분리된 읽기 전용 평가. 체크포인트, optimizer, pool, W&B를 수정하지 않습니다.

- `tournament.py`: 입력 검증, 전체 대진, 조합 단위 저장/재개, 성적표
- `policy_io.py`: 과거 184D 제출 번들의 관측·정규화 호환 연결
- `parallel_tournament.py`: 독립 GPU 프로세스에 대진을 배정하고 기존 실행기로 결과 검증·저장
- `cuda_fdm/search_eval.py`: 공통 GPU 경기 실행 및 mirrored-block 통계
- `test_*.py`: 순위 집계, 재개, 시나리오 분리, 번들 호환 회귀검증

저장소 루트에서:

```powershell
python evaluation/tournament.py --spec evaluation/models.template.json --output ../evaluation_results/my_league --validate-only
python evaluation/tournament.py --spec evaluation/models.template.json --output ../evaluation_results/my_league
python -m unittest discover -s evaluation -p "test_*.py"
```

템플릿의 models를 채워야 합니다. `scenario`는 `three_nine` 또는 `headon`이며 혼합하지 않습니다.
Head-on 거리는 `headon_distance_m: 5539.0`으로 명시합니다.
각 시나리오는 별도 출력 폴더를 사용합니다. 기본 50판 = 25개 초기조건 × 역할 교환 2회입니다.
행동은 argmax, 시뮬레이션은 CUDA/10Hz/substeps6/200초, 경기 고도 하한은 304.8m입니다.

일반 모델:

```json
{"name":"main_70000", "path":"C:/absolute/path/iter_70000.pt"}
```

archive 정책은 `config_from`에 같은 구조의 전체 체크포인트를 지정합니다.
184D 제출 번들은 검증된 관측 계약에 한해 다음과 같이 사용합니다.

```json
{"name":"submission4499", "kind":"legacy184_bundle", "path":"C:/absolute/path/iter4499_bundle", "legacy_min_altitude_m":300.0}
```

4499의 관측은 214D에서 공유 164개 특성과 마지막 행동 이력 20개를 선택합니다.
고도 여유 특성은 원래 300m 기준으로 되돌리고, 원본 float64 정규화/clip10 및 가중치를 유지합니다.
이는 경기 하한을 변경하는 것이 아닙니다. 다른 184D 모델이 같은 의미론이라고 자동 가정하지 마세요.
GPU 실행기는 BT를 지원하지 않습니다. BT/MPC CPU 평가는
[NATIVE_BASELINES.md](NATIVE_BASELINES.md)를 참고하세요.

결과: `manifest.json`, `pairs/*.json`, `progress.json`, `rankings.csv`, `score_matrix.csv`, `report.json`.
입력/메타데이터/실행기 해시와 시나리오·거리가 다르면 기존 결과와 혼합하지 않습니다.
`--stop-file PATH`가 존재하면 완결된 조합 경계에서 중지합니다.

평균 점수는 (승+0.5무)/경기수입니다. 최저 상대 점수와 추락률도 제공합니다.
관측 순위는 상대집합과 표본에 의존합니다. 최종 선발은 새 seed로 상위권을 재검증해야 합니다.
`smoke.headon.json`은 로컬 파일을 참조하는 기능검증용 2판 설정이며 성능 평가용이 아닙니다.

병렬 평가를 사용하려면 같은 명단으로 먼저 GPU 비교를 수행합니다.

```powershell
python evaluation/parallel_tournament.py --spec evaluation/campaign.headon.json --output ../evaluation_results/headon_benchmark --benchmark-only
python evaluation/parallel_tournament.py --spec evaluation/campaign.headon.json --output ../evaluation_results/headon --benchmark-result ../evaluation_results/headon_benchmark/benchmark.json
```

4대진을 1·2·4프로세스로 예열 후 비교합니다. 경기별 모든 기록과 통계가 정확히 일치하고
측정 시간이 10% 이상 줄어든 방식만 선택하며, 그렇지 않으면 1프로세스를 사용합니다.
각 프로세스는 별도 환경·RNG를 가지며 한 대진의 50경기는 그대로 유지합니다.
부모만 결과를 검증하고 저장합니다. 실행기와 worker 수도 manifest에 포함하므로
다른 방식으로 실행한 결과와 같은 폴더에서 섞지 않습니다.
이 소규모 확인은 모든 정책·하드웨어에서의 비트 단위 동등성을 증명하지는 않습니다.

`start_after_league.ps1`은 앞선 평가 프로세스 종료 및 완료 보고서를 확인한 뒤
GPU 비교 → 선택된 방식으로 다음 평가를 순서대로 실행합니다.
비교는 다른 GPU 작업이 끝난 뒤 해야 처리속도를 제대로 판단할 수 있습니다.
