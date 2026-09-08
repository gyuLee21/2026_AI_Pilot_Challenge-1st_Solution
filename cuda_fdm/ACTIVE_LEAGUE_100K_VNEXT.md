# Active League 100K vNext runtime

최종 갱신: 2026-09-05

이 런타임은 20K 체크포인트를 이어받는 준비 패키지가 아니라, 214차원 관측과
보조 예측 head를 사용하는 **scratch 100K 캠페인**의 실제 실행본이다. 현재
진입점은 상위 디렉터리의 `launch_100k.sh`이며 `cuda_fdm.train_gpu`를 직접
실행한다. 과거 coordinator/receipt/stage-0 설명은 현재 실행 계약이 아니다.

현재 three_nine 학습은 9,637 clean checkpoint에서 상승 hunter와 불용
`start_alt` 버퍼를 제거한 뒤 재개했고, iteration 10,000 clean 경계에서
`archive_only` 직접 재승격 버그를 수정했다. historical 복귀 gate를 Main
UCB95 0.65 이하로 마이그레이션한 뒤 기존 W&B run `z655uofj`로 10,001부터
online resume했다. iteration 11,473 clean 경계에서는 post-side 승인 정책이
다음 milestone까지 active pool에 들어오지 못하는 연결 누락을 수정하기 위해
중단했다. archive 131 altitude sentinel은 archive에는 probationary로 승인됐지만
active/solver에서 빠져 있었으므로 재개 전 자동 복구 대상으로 고정했다. 로컬
`metrics.csv`와 W&B 양쪽을 모니터링한다. 복구는 성공해 같은 iteration 11,473
checkpoint로 재저장됐고, active `1/4/15/3=23`, solver 19 상태에서 재개했다.
11,500에는 일반 exploiter 137이 admission을 통과했으며, full resident pool의
latest/current snapshot 표현을 abstract proposal이 deduplicate해 첫 commit은
명시적 pending으로 안전하게 실패했다. 고정 latest/recent와 strategic target의
검증 경계를 바로잡은 뒤 137을 공통 resume recovery로 활성화했다. iteration
11,600 저장본은 active `1/4/16/3=24`, solver 20, 중복/stranded 0이며 137은
challenger 1,513 games, 131은 core 1,491 games 실제 노출 상태다. W&B `z655uofj`와
함께 계속 학습 중이다.

## 현재 고정 계약

- 시나리오: `three_nine`와 `headon`을 완전히 분리한다.
- 순서: `three_nine` 100,000 완료 후 `headon`을 새 run root에서 시작한다.
- 모델: MLP `384,384,384`, auxiliary prediction ON, 214D protocol.
- main: 4,096 env, 100,000 iterations, rollout 64 고정.
- vNext: scratch `source_iteration=0`, staged mode, stage 1 live adapter.
- milestone: 500 iterations, active cap 24.
- active 역할 상한: latest 1, recent 4, core 16, challenger 3.
- admission: confirmatory score 0.55 이상 및 LCB95 0.50 이상. 같은
  role/profile successor는 동일한 frozen-main evidence에서 candidate LCB95가
  incumbent UCB95보다 높아야 한다.
- 새 challenger는 probationary로 들어오며 active exposure 32 games와 이전
  solver 포함 이력을 만족한 뒤 solver-eligible이 된다.
- stale 퇴장은 main EMA win rate 0.95 이상이면서 2,000 games 이상일 때
  milestone당 최대 2개다. archive record와 policy bytes는 삭제하지 않는다.
  퇴장된 record는 milestone마다 순환 재감사(chronological_periodic_fifo_v1)
  대상이며, main 대비 UCB95가 재진입 문턱(`historical_counter_main_ucb_max`
  0.65) 이하이면 challenger probation으로 자동 재진입한다. `archive_only`는
  일반 milestone-main 승격 후보에서 제외되므로 이 audit을 우회해 직접 core로
  돌아갈 수 없다. 복귀 후 active 노출 32게임, solver row와 Nash 순위를 다시 거친다.
- rollout이 늘어나지 않는 것은 `--rollout 64`와
  `--sched-rollout-cap 64`가 같은 값인 의도된 설정이다.
- exploiter는 `altitude_hunt`(하강 유도) 고정 슬롯 하나를 갖는다. iteration
  1000, 6000, 11000, ... (주기 5000)에 실행되며, screening/confirmatory 없이
  pool에 직접 진입한다. 여기서 생략하는 것은 admission screening이며, active
  solver의 완전성과 Nash 일관성을 위한 payoff row는 같은 milestone에 채운다.
  이후 evict는 다른 challenger와 동일한 Nash 랭킹을 따른다. 상대 log-고도
  하강량 × `alt_hunt_coef`(기본 5.0)가 geometry를 대체하고, damage와 terminal
  reward는 유지된다.
- side learner에서 승인된 일반 exploiter와 altitude sentinel은 같은 milestone에
  solver row 완성, Nash 재계산, roster 동기화까지 끝낸 뒤 다음 Main rollout부터
  challenger로 사용한다. active 24/solver 20이 꽉 찼으면 새 후보의 challenger
  좌석을 우선 확보한 **최종** `1/4/16/3` target roster를 먼저 계산·검증하고,
  기존 roster와의 diff를 한 번에 반영한다. add-then-remove 중간 상태는 만들지
  않는다. 제외된 policy bytes와 archive record는 삭제하지 않는다.

## 3-9에서 head-on으로 넘어갈 때

런처는 두 시나리오의 저장소와 checkpoint 혼입은 막지만, 순서를 자동으로
강제하거나 head-on을 자동 시작하지 않는다. 다음 조건을 사람이 확인해야 한다.

1. `three_nine`의 `metrics.csv` 마지막 iteration과 `checkpoint.pt`의 저장
   iteration이 모두 100,000이다.
2. iteration 100,000의 exploiter/evaluation/roster commit까지 끝났고
   `cuda_fdm.train_gpu` 프로세스가 더 이상 없다.
3. head-on run root에 이전 checkpoint, metrics, league 또는
   `vnext_control/shadow_state.json`이 없다.
4. `bash launch_100k.sh headon`으로 scratch 시작한다. 3-9 checkpoint나
   league/archive/control state를 복사하지 않는다.

재개는 정확히 `bash launch_100k.sh <scenario> resume`을 사용한다. checkpoint의
`initial_condition_scenario`가 다르면 load가 거부되므로 3-9를 head-on으로
잘못 resume할 수는 없다.

## 현재 알려진 제한

- `maximum_gpu_learners=1`은 configuration 검증값일 뿐 OS process/device
  lock이 아니다. 3-9 실행 중 head-on을 호출하면 두 GPU learner가 동시에 뜰
  수 있다.
- `fresh` 경로는 기존 run root 충돌을 거부하지 않는다. 같은 시나리오를
  `resume` 없이 다시 실행하거나 mode를 오타 내면 기존 metrics/archive/control
  state 위에 새 모델을 시작할 위험이 있다. mode도 `resume` 외 값을 명시적으로
  검증하지 않는다.
- exploiter early-stop은 target iteration에 완료 episode가 있어 threshold를
  넘은 경우가 2회 연속이어야 한다. 완료 episode가 0인 target iteration이
  사이에 끼면 streak가 초기화되어, 우세한 exploiter도 최대 iteration까지 갈
  수 있다. 현재 회귀 테스트 228개 중 이 조건을 다루는 기존 테스트 1개가
  실패한다.
- core는 milestone마다 전체 재순위화되며 hysteresis가 없다. 지금은 core가
  덜 찬 상태라 용량 문제는 없지만, 16석이 찬 뒤 cutoff 근처의 작은 Nash 변동이
  churn을 만들 수 있으므로 `core_roster_diff`를 감시한다.
- content-addressed 파일은 중복 bytes를 한 파일로 저장하지만 서로 다른
  archive identity가 같은 bytes로 active seat를 각각 차지하는 것을 일반적으로
  막지는 않는다. 최근/milestone 동시 archive 중복 경로는 별도로 방지되어 있고
  이 항목은 낮은 확률의 잔여 위험이다.
- `BatchObsReward._AC_KEYS`(obs_reward.py)와 checkpoint의 stagger snapshot
  (`ckpt["runtime"]["obr"]`)은 키 계약이 같아야 한다. 과거 `start_alt`를
  추가했을 때 기존 checkpoint resume가 KeyError로 종료된 적이 있었다. 현재는
  상승 hunter와 해당 버퍼를 모두 제거하고 9,637 checkpoint의 그 키도
  제거해 해결했다. 향후 필수 runtime 버퍼를 추가할 때는 checkpoint
  마이그레이션과 resume 테스트를 같이 수행해야 한다.

## Iteration 11,473 post-side 즉시 투입 수정

- 기존 `on_milestone()`은 side learner 실행 전에 solver와 roster를 확정했다.
  side learner의 confirmatory 또는 altitude direct-entry가 그 뒤 성공해도
  `last_solver_ids`를 다시 만들지 않고 이전 roster만 동기화해, 승인된 후보가
  다음 500-iteration milestone까지 archive에만 머물렀다. 일반 exploiter 119와
  125, altitude sentinel 131에서 실제로 이 지연을 확인했다.
- 수정 후에는 신규 승인 후보 한 자리를 먼저 예약하고, 기존 probationary 공정
  대기열과 strategic 순위로 solver 20을 다시 구성한다. 신규 row의 missing edge가
  아니라 proposed solver 전체의 **실제 missing-edge set**이 completion cap 48
  이하인지 확인한 뒤 평가·Nash·roster를 원자적으로 갱신한다. 현재 11,473
  checkpoint에서 131을 포함한 solver는 19개이고 실제 missing edge는 18개로,
  cap 48 안에 들어간다.
- full pool에서도 active 역할 상한 `1/4/16/3`을 넘기지 않는다. 탈락한 기존
  probationary는 `admission_status=probationary`를 유지한 채 별도 metrics의
  `solver_status=pending`과 pending reason을 갖는다. 즉 solver-pending을 admission
  lifecycle 값으로 대신 표현하지 않는다. 진행 중 에피소드가 끝날 때까지 stable
  resident ID는 유지된다.
- resident latest는 archive id가 없고, milestone의 frozen current snapshot은
  recent에 존재할 수 있다. 따라서 post-side strategic refresh는 고정
  latest/recent 수를 실제 resident pool에서, core/challenger 수를 target proposal에서
  따로 검증하고, sync 뒤 실제 `1/4/16/3` 및 archive-id 중복 0을 다시 검사한다.
- payoff 평가, Nash 또는 roster 적재가 실패하면 기존 solver/roster를 유지하고
  후보를 명시적 pending으로 기록한다. `admitted/probationary`인데 active, solver,
  pending, evicted 어느 쪽도 아닌 상태는 milestone health assertion이 거부한다.
- pre-fix checkpoint를 resume할 때 특정 archive 번호가 아니라 일반 stranded
  detector가 조건에 맞는 모든 후보를 찾고 공통 activation 경로로 복구한다.
  payoff/Nash/roster transaction 실패처럼 재시도 가능한 pending도 resume에서 같은
  경로로 재시도하되, edge cap 초과처럼 의도적인 capacity pending은 매 launch마다
  반복 평가하지 않는다.
  복구 직후 같은 committed iteration으로 checkpoint를 즉시 다시 저장하며,
  재호출은 no-op이다.
- altitude direct-entry는 admission screening/confirmatory만 생략한다. solver
  payoff는 생략하지 않는다. 또한 confirmatory admission health counter와 분리해
  기록하므로 direct-entry가 일반 admission pipeline의 성공률을 부풀리지 않는다.
- 공정 대기열은 pending 시각뿐 아니라 challenger 활성화 횟수와 마지막 활성화
  iteration을 사용한다. 매 milestone 새 후보가 들어오는 합성 20-milestone
  테스트에서도 기존 probationary 10개가 모두 최소 한 번 선택된다.

## Iteration 10,000 pool 재진입 수정

- archive 4는 iteration 9,000 stale 퇴출 뒤 9,500 audit에서 Main UCB95 1.0으로
  재진입에 실패했는데도, `archive_only`를 허용하던 일반 past-main 승격 경로로
  잘못 복귀했다. 10,000에서는 Main EMA 0.9927로 다시 정상 퇴출됐다.
- 수정 후 일반 past-main 직접 승격은 처음 승격되는 `None/admitted` record만 허용한다.
  퇴출된 record의 유일한 복귀 경로는 UCB95 0.65 이하 audit → probationary
  challenger → 32게임 노출/solver → Nash core 선발이다.
- iteration 10,000 clean checkpoint를 별도 백업한 뒤 serialized vNext config와
  `shadow_state.json`의 gate 한 값만 0.50에서 0.65로 마이그레이션했다. 모델,
  optimizer, normalizer, pool, 진행 중 episode, payoff, archive와 RNG를 포함한
  나머지 checkpoint 상태가 모두 동일함을 재귀 비교했다.

상세 운영 절차, 검사 시점의 실제 풀 상태와 테스트 결과는
[패키지 README](../../README.md)와
[구현 상태](../../IMPLEMENTATION_STATUS.md)를 본다.
[AUTHORITATIVE_VNEXT_SPEC.md](../../AUTHORITATIVE_VNEXT_SPEC.md)는 SHA-256으로
고정된 과거 설계 스냅샷이며 현재 runbook이 아니다.
