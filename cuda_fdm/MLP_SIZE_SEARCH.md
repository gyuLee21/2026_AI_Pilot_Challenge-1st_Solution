# MLP depth-first, width-second search (auxiliary ON)

User-approved reward replacement, 2026-08-31. This supersedes earlier search plans,
not their historical results. `experiment_v1` remains held; `mlp_depth_width_aux_v1`
is stopped with its manifest, logs and checkpoints preserved. Only
`runs/architecture_search/mlp_depth_width_aux_altmix_v2` is authorized now.
No old weights, optimizer state, opponent history or rankings enter this cohort.

## Frozen sequence

1. At width **512**, compare **2, 3, 4 hidden layers**. Each trains from scratch
   with seeds **0/1/2 to 800 iterations**. Iter400 is intermediate, not a winner
   selection. Select depth by common evaluation of all nine iter800 checkpoints.
2. At that depth, compare widths **512, 768, 1024**, seed0, **800 iterations**.
   The matching width512 checkpoint is reused from step1; never retrained or
   compared at a different budget. **Only if 512 ranks first among those three**,
   test **384 once**, at the SAME depth/budget/evaluation seed, then rank all four.
   Stop expansion there. No residual/extra-layer/GRU experiment.
3. The top two widths train with seeds **0/1/2 to 1500**, resuming only checkpoints
   from this same corrected protocol. Evaluate iter500/1000 as diagnostic curves
   and **iter1500 with separate held-out evaluation seeds** for final selection.
4. Export and time the actual two-agent CPU submission pipeline. Eligible models
   require **p99 <50ms and max <100ms** at 10Hz, CPU threads1. This is local
   headroom, NOT certification of unknown network/server load.
5. Choose the highest-ranked eligible architecture and start a **fresh seed0 main
   for 20000 iterations**. No search optimizer, normalization, opponent pool or
   model weights are inherited. Convert the completed checkpoint to a bundle.

This is a bounded sequential search, not proof of a global optimum. In particular,
depth is selected at width512; depth/width interactions are not exhaustively tested.
Confidence intervals and individual training-seed results must accompany rankings.

The approved width plan `width512_768_1024_then384_v2` is retained: 768 is the
midpoint of 512 and 1024. Depth/width budgets and evaluation seeds are unchanged.
Unlike the earlier manager-only width revision, this is a reward change: all
comparisons start fresh under a NEW manifest, never an in-place source rebaseline.
The old manager exited at a safe boundary (512x3 seed2 iter403). Its sources,
manifest, STOP and that full checkpoint were copied to
`runs/change_validation/altitude_hp_altmix_v2/before` before edits. Earlier
checkpoints remain in the stopped cohort. They are not invalid as historical
observations but cannot rank models trained on this different objective.

## Shared learning conditions

- Actor/critic independent Tanh MLPs; identical tested hidden layout for both.
- **Adam retained**: eps1e-5, no weight decay. Do not substitute AdamW during search.
- 184D observation, control10Hz, physics60Hz, substeps6.
- 4096 environments, rollout64 (262144 main transitions/iteration), epoch4,
  minibatch **splits8** (not batch size8), lr3e-4, entropy.001, clip.2, KL target.03.
- Gamma.997, lambda.95, final-safe geometry 50/20/20/10 with trapezoid and +/-5
  budget unchanged. Remove own low-altitude/descent shaping entirely. On an
  altitude exit, own reward is -remaining HP x10, opponent reward +remaining HP
  x10, using POST-damage HP so the last frame is not double-counted. Existing
  own-altitude-first priority for simultaneous crashes is retained. HP kill/loss
  +/-5, timeout +5/-5/draw-4 and damage differential x10 are unchanged.
  3-9:head-on=3:1 and the official 1000ft (304.8m MSL) termination threshold are fixed.
  The 200s limit remains a terminal finite-horizon objective (no value bootstrap).
- Auxiliary ON, coefficient **0.1** for all candidates. Contract is
  `cuda_mlp_flat_finite_horizon_future_aux_altmix_v6`; see `FUTURE_AUXILIARY.md`.
- Actor loss = clipped PPO loss - entropy coefficient * entropy + .1 masked MSE.
  Reward enters PPO through advantages/returns; it is not directly added to MSE.
  Auxiliary heads are removed from the submission bundle; controls remain four.
- PFSP gated EMA-softmax unchanged: eviction cap4, gate.6, EMAalpha.1, temp.3,
  uniform floor.5. Opponent identity/normalization are fixed until each episode ends.
- Search: milestone0, exploiter0, schedule0. **No W&B**. All scalar CSVs, metrics,
  model/optimizer/runtime checkpoints and game-by-game evaluation records are local.
- No 1220/4499, old historical references, BT/MPC, AlphaStar, new observations,
  further reward retuning, coefficient or optimizer search. Comparison exploiters
  remain OFF; alternating exploiters are enabled only in the final main.

The auxiliary coefficient is a conservative initial choice, not established as
optimal. A read-only 4096x64 actor-trunk diagnostic measured weighted-aux/PPO+entropy
gradient norm ratios of ~0.028% initially and ~0.015% after two updates. This is
early-training evidence only. MSE is normalized by 100m; 0.1 is NOT a 10% PPO share.
Adam preconditioning means raw gradient ratios are not parameter-update ratios.
Keep the coefficient fixed throughout this cohort; monitor MSE versus CV baseline
and actual combat evaluation instead of silently tuning it for a favored model.

## Evaluation and ranking

Each matchup has **256 complete games per scenario**, organized as 128 identical
initial-state pairs with aircraft roles exchanged. Separate three-nine and head-on
results are weighted **75:25**. Policies are deterministic, using their own frozen
normalization. No short-episode early finish sampling bias; all lanes must finish.

- Common references for depth and width: **all three depth seed0 iter200 policies**,
  a set chosen before observing results; never remove eliminated references.
- Final references additionally include **every screened width seed0 iter400**,
  including 384 if it was triggered, even when a width is eliminated.
- Cross-play includes every cross-family training-seed pairing. Aggregate per
  model/seed first, then average seeds equally.
- Selection: **75% cross-play +25% fixed-reference score** (draw=.5).
  Ties: actual win rate, damage differential, fixed-reference score.
- Training returns/PFSP win rates are diagnostics, not the selection objective.
- Seed sets: depth82001, width83001, curve83011, final95001.
- Paired-game bootstrap CIs and three-training-seed mean CIs remain distinct.
  An observed higher mean with overlapping CIs is not a proven superiority claim.

Evaluation must have the exact expected scenario/match/game counts, finite scores,
valid mirrored role indices and bounded times. Checkpoint hashes are frozen per
suite. Partial suites, invalid trajectories and foreign checkpoints cannot rank.

## Main and stopping

Main restores the existing schedule: every2000 iter, LR/entropy x1/3, rollout+8,
shaping ladder 1/.6/.32/.12/0. Permanent main snapshot every500 iter. Run ONE
exploiter at each milestone: altitude_hunt at500, standard at1000, altitude_hunt
at1500, standard at2000, repeating up to20000 (20 of each type). Each trains for
max1000 iterations or win-rate EMA>=.75, against a frozen current main.
The existing first500/rest1000 initialization snapshots, PFSP and pool insertion
are retained. A hunter is not added twice and does not add new milestone work.

Hunter reward follows the scoped idea in Git ad632723: replace geometry with
`5 * (log(target_alt_prev/304.8) - log(target_alt_next/304.8))`, with log input
clamped at1e-4 for finite arithmetic. Target descent is positive, climb reverses
it, level flight is zero. Damage and terminal rewards still apply. Initial logs
come from the initial episode state, never the previous episode. Main/standard
reward has NO such term. The mode is derived from the main milestone index, not
a counter which can drift after resume. Main model, optimizers, live episodes,
reward mode, reconstruction, opponent IDs and RNG restore even on exceptions.
Mode, target crash rate and achieved EMA are logged locally and on main W&B;
each completed exploiter's provenance is included in recovery checkpoints.

This does not prove intentional causation of a crash: naturally descending main
policies can also earn hunter reward. Low hunter loss/high return alone is not
success. Track main crash rate and independent combat evaluation. The user's
reported BT10:0 result from another run is motivation, not a reproduced result here.
Main seed0 is fresh. The existing 2000-iteration schedule is NOT stretched when
extending from10000 to20000; ordinary iteration cost grows as rollout length does.

Recovery model/optimizer/runtime saved every100 iter; named milestone snapshots
every500. No checkpoint upload. Main W&B only:
`leeai021213` / `leeai021213-ajou-university` / `AIP contest`.
Wrong/missing credentials fail closed; never use `leetm2021` or silently run main offline.

Run:

```powershell
python -B -m cuda_fdm.mlp_size_search
```

`STOP` in the NEW cohort stops at a completed main iteration or evaluation match.
A milestone's exploiter may finish before the main iteration boundary is reached.
NaN/Inf, source/config mismatch, failed preflight or incompatible bundle blocks
the cohort with `INTEGRITY_FAILURE.json`; never erase/rebaseline automatically.
The old experiment's HOLD/STOP are not bypassed: that experiment is never resumed.

A single-GPU manager lock and prior-child PID check reject duplicate runs. Normal
process interruption can resume from matching atomic recovery checkpoints after
checking for orphans. Uncheckpointed CSV tail is archived in full before replay,
not counted twice. Invalid/numerical failures are never automatically retried.

Status, current child PID/log, recent iteration time and current-job ETA update
every30 seconds. GPU telemetry is local. Report main ETA separately from additional
exploiter time: the ordinary iteration timer does NOT include milestone exploiter
duration. Do not advertise its sum as full wall-clock runtime.

The existing Codex `cuda` heartbeat monitors this manager, validates completed
stages and reports significant changes. The manager dispatches the next approved
stage itself, without waiting for a heartbeat. On final20000 + bundle completion,
report paths/results/W&B link and pause the heartbeat. STOP and later user requests
override automatic progression. Computer must remain awake for local training.
