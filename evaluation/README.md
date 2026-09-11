# GPU 리그전

학습과 분리된 읽기 전용 평가. 체크포인트, optimizer, pool, W&B를 수정하지 않습니다.

- `tournament.py`: 입력 검증, 전체 대진, 조합 단위 저장/재개, 성적표
- `policy_io.py`: 과거 184D 제출 번들의 관측·정규화 호환 연결
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
BT는 지원하지 않습니다.

결과: `manifest.json`, `pairs/*.json`, `progress.json`, `rankings.csv`, `score_matrix.csv`, `report.json`.
입력/메타데이터/실행기 해시와 시나리오·거리가 다르면 기존 결과와 혼합하지 않습니다.
`--stop-file PATH`가 존재하면 완결된 조합 경계에서 중지합니다.

평균 점수는 (승+0.5무)/경기수입니다. 최저 상대 점수와 추락률도 제공합니다.
관측 순위는 상대집합과 표본에 의존합니다. 최종 선발은 새 seed로 상위권을 재검증해야 합니다.
`smoke.headon.json`은 로컬 파일을 참조하는 기능검증용 2판 설정이며 성능 평가용이 아닙니다.
