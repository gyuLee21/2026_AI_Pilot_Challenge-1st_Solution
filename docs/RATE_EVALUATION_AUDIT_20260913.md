# 10Hz / 60Hz 평가 감사 — 2026-09-13

## 결론

동일한 36k 모델에서 GPU 10Hz 70승/60Hz 30승, CPU 10Hz 64승/60Hz 36승은 저장된 결과와 일치한다. CPU 기록 100판 모두 실제 호출 횟수와 결정 횟수가 10:60 비율을 만족한다. 그러나 이를 과거 Head-on 46k와 비교해 순수한 모델 특성으로 설명할 수는 없다. 모델·시나리오·시드 구성·초기 관측이 다르다.

우리 평가와 이태민 원본은 완전히 동일하지 않다. 특히 원본 high_rate의 관측용 HP/연료/시간 적분은 10Hz이고, 우리 DecisionRateProvider는 60Hz다. 실제 물리 HP/승패 계산과 모델 내부 재구성 HP는 별개이며 혼동하면 안 된다.

이번 작업은 코드·기록 검토와 고정 입력 재생 검사다. 이태민 구현으로 100판을 새로 돌린 결과나 관측 차이가 승패 반전의 원인임을 입증한 실험은 아니다. 학습·제출 실행 파일은 변경하지 않았다.

## 원본 확보

- 원본: https://github.com/idearendil/AIP/tree/c3374d4d5c57d8855e1f7276149d96e0479d1e6a
- 기존 감사 clone에서 origin을 새로 fetch. origin/main2는 c3374d4 유지.
- 주요 수정: ce38736fe031a4a4db44d9635663d388225b884c (`fixed bugs with 60hz power test`).
- 수정하지 않은 원본 11개 파일: workspace `work/taemin_eval_reference_complete_20260913/`.
- 원본 ZIP: `work/taemin_eval_reference_complete_20260913.zip`.
- 주요 파일: high_rate.py, power_test.py, evaluate.py, final_power_test.py, mlp_vs_lstm_test.py, action_provider.py, my_observation.py, env_utils.py, model.py, single_agent_env.py, FighterSim.py.

## 평가 경로 비교

|항목|이태민 원본|현재 CPU 주기 비교|현재 GPU 주기 비교|
|---|---|---|---|
|모델|원본 테스트 기본 번들/상대는 우리 36k와 다름|동일 frozen 36k 두 개|하나의 frozen 36k를 양쪽 배치에 사용|
|샘플링|power_test ownship은 stochastic; target은 옵션으로 deterministic, high-rate 선택 가능|양쪽 argmax|양쪽 argmax|
|10Hz 행동|_ActionRepeatProvider 6프레임 유지|6프레임 유지|6프레임 유지|
|60Hz 행동|HighRateProvider 매 프레임 재결정|매 프레임 재결정|매 프레임 재결정|
|관측 기하|매 프레임 신선한 상태|동일|동일|
|60Hz pqr/가속도|현재-0.1초 전 sliding 차분|동일|동일 수식, CPU 함수와 수치 검사 통과|
|60Hz 행동 이력 시차|0.1,0.2,...0.5초|동일|동일|
|관측 HP/연료/시간|0.1초 경계에서 dt=.1 적분|매 프레임 dt=1/60 적분, 10Hz 정책도 동일|매 프레임 dt=1/60 적분|
|순간 피해 관측|60Hz 현재 기하, 시간은 10Hz 경계 값|60Hz 현재 기하 및 시간|60Hz 커널|
|스로틀 이력|command 변환 전 raw [-1,1]|GPU checkpoint용 적용 command [0,1]|훈련 커널과 같은 적용 command [0,1]|
|LSTM 상태|60Hz 예측하되 hidden은 10Hz 경계 commit|매 결정 hidden 갱신|MLP만 허용|
|초기 속도|FighterSim reset 반환은 0|수정된 FighterSim의 실제 FDM 속도|요청한 IC의 속도|
|물리 피해 적분|CPU 환경에서 매 60Hz substep|원본과 같은 CPU 환경 루프|이번 rate runner는 매 60Hz; 일반 GPU 평가기는 10Hz 끝점 적분|
|종료 확인|CPU 6 substep 뒤 일반 종료 확인|동일|매 1 substep 확인|

MLP인 36k와 46k에는 LSTM hidden 차이가 적용되지 않는다. action history의 raw throttle 규약을 원본에서 무조건 복사하면 GPU 학습 checkpoint의 입력 계약과 어긋난다. 현재 GPU obs_kernel은 `actions[a*4+c]`를 그대로 history에 넣으며 이 actions의 throttle은 [0,1]이다.

CPU single_agent_env.py는 가져온 원본과 diff가 없다. 두 provider 모두 저장된 이전 substep 상태를 받으며, ownship 물리 업데이트 후 target provider가 호출돼도 target에 넘기는 저장 상태는 같은 시점이다. 단 CPU 루프의 일반 종료 확인은 0.1초 단위이고, 이번 GPU rate loop는 1/60초 단위이므로 동시 파괴/동시 추락 처리 수치는 달라질 수 있다.

원본 final_power_test는 worker 실패 경기를 제외해 보고한다. 우리 평가의 정상 완료 100판과 실패 제외 숫자가 있는 원본 결과를 같은 표본이라고 취급하면 안 된다. 원본 기본 시나리오 설정도 우리 강제 3-9/headon 설정과 같다고 가정할 수 없다.

## 원본 high_rate의 핵심 차이

원본 high_rate.py 주석은 60Hz로 HP/연료/시간을 적분해 시험했으나 승률이 떨어져 0.1초 경계 적분으로 되돌렸다고 명시한다. 이 주석은 원저자의 경험 보고이지 우리 36k 결과에 대한 인과 증거는 아니다.

정확한 원본 HighRateProvider 클래스를 가져와 같은 로컬 관측 함수와 36k frozen actor를 연결하고 동일한 합성 접근 상태 121프레임을 재생했다. 이를 통해 provider 자체 의미만 분리했다.

|프레임|원본 관측 시간|현재 관측 시간|원본 추정 HP|현재 추정 HP|
|---|---:|---:|---:|---:|
|1|0|0.016667|1.000000|0.999539|
|5|0|0.083333|1.000000|0.996238|
|6|0.1|0.1|0.992861|0.995048|
|60|1.0|1.0|0.692388|0.714261|

121프레임 중 실제 command가 다른 프레임은 1개였다. 이 작은 고정 입력 재생은 의미 차이를 증명하지만, 폐루프 대전 승률에 미치는 크기는 증명하지 않는다. 이력 스로틀도 원본 raw 0.8 / 현재 mapped 약0.9처럼 서로 다른 척도임을 확인했다.

## 과거 Head-on 평가 코드 해시 차이는 해소됨

과거 manifest에 기록된 decision_rate.py SHA256:
`240621edab6082abfe727fde00a44cc675ff843f196b5344cb7c55056b142d1c`.

동일 해시의 파일을 `releases/headon46k_60hz_20260912/build-input/submission/decision_rate.py`에서 찾았다. 현재 파일과 유일한 코드 차이는 이산 행동 인덱스의 유효성 검사와 `.long()` 변환이다. FrozenPolicy가 반환하는 int64 argmax에는 의미 변경이 없다.

같은 36k actor, 동일한 121개 상태에서 과거/현재 provider 비교:
- 10Hz: 각 21회 결정, 모든 command 차이 0.
- 60Hz: 각 121회 결정, 모든 command 차이 0.
- frozen_policy.py 및 my_observation.py는 과거 manifest와 현재 SHA256 동일.

따라서 이 해시 차이를 46k/36k 승패 반전의 원인처럼 남겨 두는 것은 부정확하다.

## 초기화는 실제로 다름

원본 FighterSim과 현재 CPU runtime의 유일한 실질 diff는 reset에서 `get_fdm_data()`와 `_update_state()`를 호출해 실제 FDM 상태를 반환하는 추가 코드다.

- 과거 headon 46k block_000 초기 속도: 두 기체 모두 [0,0,0].
- 현재 36k CPU block_000 초기 속도: 전방 약222.5104m/s, 작은 v/w 포함.
- 실제 서버 이번 캡처 frame0 속도: 두 기체 모두 [0,0,0]. frame1 약200.15m/s.

따라서 현재 CPU full-state reset이 서버의 첫 직렬화 관측을 완전히 재현한다고 말할 수 없다. 반대로 초기 속도만 0으로 되돌리면 전체 위치/시점까지 서버와 같아진다고도 말할 수 없다. 서버 INIT는 이번 캡처에서 받지 못했다.

## 저장 결과와 통계 해석

|조건|10Hz 승|60Hz 승|무승부|표본|
|---|---:|---:|---:|---|
|과거 46k Head-on CPU|25|38|37|50개 seed 각각 좌우 교환, 100판|
|36k 3-9 GPU|70|30|0|독립100 seed, 기체 슬롯/방향 균형|
|36k 3-9 CPU|64|36|0|같은100 seed, 실제 시작 방향 균형|

과거 46k의 10Hz score는 .435이고, 저장된 paired bootstrap95는 [.355,.520]으로 .5를 포함한다. 60Hz 관측 승수 우세를 확실한 통계적 우세나 모든 모델에 대한 일반 법칙으로 확대하면 안 된다.

36k CPU: 시작 방향별 10Hz 승수 29/50, 35/50. 모든 경기의 provider 호출수는 양쪽 동일하고 60Hz 결정수가 10Hz 결정수의 정확히 6배. 100개 seed 중복 없음. 추락은 10Hz 25판, 60Hz 58판. GPU 추락은 각각 7판,19판이므로 CPU/GPU 경기 분포의 차이는 여전히 크다.

## 권장되는 최소 후속 검증

1. 현재 결과는 해당 평가 설정에서 유효한 관측으로 보존한다. 60Hz가 보편적으로 열등하다거나 유리하다고 결론내리지 않는다.
2. 폐루프 36k 재평가를 한다면, 초기화와 argmax·가중치·시드는 고정하고 관측 HP/연료/시간만 원본의 0.1초 경계 적분으로 바꾸는 한 가지 A/B가 우선이다. GPU 학습에 맞는 [0,1] throttle history와 기존 pqr/가속도 sliding 창은 유지한다.
3. 초기 zero/full-state 효과는 별도 실험이다. 위 A/B와 동시에 바꾸면 원인을 분리할 수 없다.
4. 실제 제출 선택은 같은 실행 계약과 실제 서버 확인에 기반해야 한다. 이번 감사만으로 재학습·제출 실행 파일 교체는 하지 않는다.

검사 재현: `work/audit_rate_contracts_20260913.py`. 수치 출력: `work/rate_contract_audit_20260913.json`.
