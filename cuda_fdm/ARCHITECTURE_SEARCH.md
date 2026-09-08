# Approved 10 Hz architecture search

## Current authorized search: MLP depth then width, auxiliary ON (2026-08-31)

The latest user instruction authorizes a **new** cohort, not resumption of the
held historical experiment below. See `MLP_SIZE_SEARCH.md` and run
`python -B -m cuda_fdm.mlp_size_search`.

The new root is `runs/architecture_search/mlp_depth_width_aux_v1`: depth2/3/4
at width512 (3 seeds x800), then width384/512/672/832 at the selected depth
(seed0 x800; conditional1024 only if832 wins), then top2 x3seeds x1500,
actual CPU two-agent latency, and a fresh10000-iteration main. Adam and auxiliary
coefficient0.1 remain fixed. Comparison W&B is off; only main uses the allowed account.

`experiment_v1` STOP/HOLD/manifest and old results stay intact. Do not mix its
auxiliary-OFF or pre-integrity-fix results into the new ranking.

## Latest implementation: future-position auxiliary learning (2026-08-31)

The user approved the auxiliary task, using the Git implementation as the
reference where sound. New training CLI runs enable the training-only six-output
heads on each MLP (opponent +0.5s and own +1.0s position residual). Observations,
controls, reward, discount, episode-fixed pool semantics and flat PPO batching
remain unchanged. This changes the training loss, not the inference interface.
See `FUTURE_AUXILIARY.md` and `runs/change_validation/future_aux_v1/`.

Do not inject this loss into the historical architecture cohort. Its manager
commands now explicitly pass `--no-aux-pred` and reject auxiliary-ON checkpoints;
its STOP, integrity hold and manifest are NOT released or rehashed. No search or
main run is launched by this implementation. A future corrected search manifest
must explicitly state its auxiliary setting, uniformly for every compared model.
Do not combine ON and OFF learning results as an architecture-only comparison.

The latest discussion requests depth **2/3/4** and considers depth-first screening
before width. That newer discussion supersedes the earlier depth-3/4-only proposal
below, but has not been converted into a newly activated execution manifest here.

## Latest user decision: MLP selected (2026-08-31)

The user accepted the historical MLP advantage as a practical model-family
decision. **Do not repeat GRU vs MLP training.** Width/depth selection remains
open. The old schedule below documents the original experiment, not permission
to restart that full search. See
`runs/architecture_search/experiment_v1/changes/mlp_selected_v1/decision.json`.

The CUDA updater is now MLP-only: flat transition shuffling, no TBPTT splits,
padding or rollout hidden-state buffers. Protocol `cuda_mlp_flat_finite_horizon_v4`
explicitly separates the different minibatch ordering from the old sequence
updater. PPO equations, hyperparameters, reward, observation and PFSP are not
retuned. GRU model loading, bundle export and evaluation remain available, but
the training CLI accepts only MLP. Existing v2/v3/legacy training checkpoints
are not silently resumed under v4.

The user subsequently approved assigned-row opponent inference. Active MLPs now
normalize/forward only their assigned environments, while preserving full-shaped
sampling RNG, original PFSP, episode-fixed IDs, retirement and checkpoint state.
77 CPU tests and real 4096-env A/B/save-resume checks passed. In the six-opponent
bounded smoke, mean iteration time fell from 3.63 to 3.00 seconds with identical
actions/targets/final parameters. See `pool_inference_fixes_v1/REPORT.md` under the
held experiment. This does not resume that experiment or select a width/depth.
The latest width/depth proposal is recorded separately in that folder's
`MLP_SIZE_DEPTH_PROPOSAL.md`: widths 384/512/672/832, **each at depths 3 and 4**
(eight initial candidates). This replaces the earlier best-width-only depth test.
If an 832-width candidate leads the common screening and passes latency checks,
add 1024 at both depths under the same conditions (ten candidates after that
conditional extension). Any further width extension needs an evidence-based
increment/budget decision, not an unbounded preset sweep. The user approved this
search direction; it is not yet an activated, corrected-cohort execution manifest.

The held original manifest/STOP/hold remain unchanged in purpose. A distinct
corrected-protocol size/depth cohort is still required; do not run the old manager
command below against the changed source or rebaseline it automatically.

The MLP extension below is implemented but **not started**. The current experiment
remains stopped under `STOP` and `INTEGRITY_HOLD.json` while the observation/submission
and opponent-identity audit is reviewed. Neither marker nor the existing manifest's
source hashes should be cleared or refreshed automatically. Existing results are
preserved as evidence; results from a repaired protocol must not be silently mixed
with the original protocol. Resume needs a reviewed protocol/provenance decision.

Launch from the repository root with the existing `aip` Python environment:

```powershell
python -u -m cuda_fdm.architecture_search --root runs/architecture_search/experiment_v1
```

The manager runs one GPU job at a time. It verifies a regression gate and real 4096-env
two-iteration smoke tests before the experiment. Only the final 10000-iteration run uses the verified W&B
username `leeai021213`, entity `leeai021213-ajou-university`, project `AIP contest`.
Regression, smoke and all comparison jobs are local only (user update 2026-08-31).
CSV, evaluation JSON and checkpoints remain fully recorded. No checkpoint artifact uploads are requested.

## Schedule

1. GRU `512-512-GRU256-512` versus MLP `672-672-672`: seeds 0/1/2, 800 iterations.
2. Winning family, seed 0, 400 iterations per candidate:
   - MLP: `384x3`, `512x3`, `672x3`, `512x2`.
   - GRU: `384-384-GRU128-384`, `512-512-GRU128-512`, current model,
     `512-GRU256-512` (encoder_depth=1).
3. Top two configurations, seeds 0/1/2, extend/train to 1500 iterations.
4. Paired held-out evaluation, CPU two-agent latency gate and bundle conversion.
5. Fresh seed-0 10000-iteration run with the selected configuration, original
   500-iteration main snapshots/exploiter, max exploiter 1000, EMA threshold 0.7.

Search jobs retain gated EMA PFSP but disable milestones/exploiters/schedules. Final
jobs restore the existing schedule. Observation=184, gamma=.997, lambda=.95,
substeps=6 (10Hz policy/60Hz physics), nenv=4096, rollout=64, epochs=4,
num_minibatches=8. No 20Hz, AlphaStar, old 1220/4499, BT or MPC additions.

## Optional MLP width/depth extension (implemented, pending activation)

This path is opt-in. An absent `manifest.json` field `mlp_extension` retains the
original schedule above. Activation requires the complete frozen `MLP_EXTENSION`
object from `cuda_fdm/architecture_search.py`; partial or changed configurations
are rejected. The manager never adds that field or refreshes an existing manifest.
The current experiment has not been opted in or resumed by this implementation.

The extension requires the preserved architecture decision `mlp` and original
size-screen top two `mlp672`, `mlp512_d2`. It does not rerun, rewrite or replace the
original `architecture_800`, `size_400`, or their decisions.

1. Add `mlp768` (`768x3`) and `mlp672_d4` (`672x4`), with the existing activation,
   actor/critic structure and training hyperparameters. No residual connections.
2. Train each new candidate with seed 0 to iteration 800. Keep iteration 400 only
   as an intermediate snapshot; do not use it as a new 400-iteration selection stage.
3. Compare the two new iteration-800 checkpoints against the existing
   `mlp672/seed0/iter_00800.pt` in `mlp_extension_800`, evaluation seed **74001**.
   All three use the original common reference set: preserved GRU100 and the two
   original architectures' seed0/iter200 checkpoints. Scenario weights remain
   75% three-nine / 25% head-on; scoring remains 75% cross-play / 25% references.
4. Select only the best **new** candidate. It joins the preserved `mlp672` and
   `mlp512_d2`; train/extend those three configurations with seeds 0/1/2 to 1500.
5. Rank all nine checkpoints in the distinct `confirmation_expanded_1500` phase,
   held-out seed **94001**. Its frozen references include the original common set,
   all original size-screen iteration-400 snapshots, and **both** new candidates'
   iteration-400 snapshots, including the eliminated new candidate.
6. Record diagnostic curves as `confirmation_expanded_curve_500` and
   `confirmation_expanded_curve_1000` with that same expanded reference set. They
   never replace the original curve phases or affect final ranking.
7. Only after the expanded confirmation completes may CPU latency checks, winner
   selection and the fresh seed-0 10000-iteration final run proceed. The final run
   does not resume a search-training checkpoint.

`mlp_extension_decision.json` records the new winner without replacing either
original decision. Existing final confirmation results remain under their original
phase name. The integrity hold currently prevents every command in either path.

## Evaluation and selection (fixed before results)

Each scenario uses 128 initial-condition pairs, with policy roles swapped on the
same ICs: 256 **complete** games, one per lane, not the first 256 completions.
All observations use the evaluated checkpoint's own frozen normalization. Actions
are deterministic. Training does not consume evaluation data.

Scenario weights: 75% three-nine / 25% head-on. Selection score: 75% cross-play
against competing configurations / 25% frozen common references. Architecture
references are preserved GRU100 and each architecture's seed0/iter200; size final
references additionally retain **all** size-screen iter400 candidates, including
eliminated ones. Within a category all opponents have equal weight. Ties use raw
win rate, damage difference, then reference score. Pair bootstrap intervals and
three-training-seed mean t-intervals are reported separately; these are uncertain
finite-sample estimates, not proof of global optimality.

Final held-out seed=94001; selection seeds=72001/73001. GRU hidden-reset experiments
are diagnostic and excluded from rankings. No strong historical policies are used.

CPU latency measures two local pipelines (reconstruction, observation, normalization,
policy, control conversion and JSON) with one Torch CPU thread. Gate: p99<50ms and
maximum<100ms, leaving headroom within a 100ms policy period. This does **not** measure
network/server latency or certify an undocumented competition-server deadline.

## Progress, resume and stop

- `status.json`: current phase, manager/child PID, log paths.
- `manifest.json`: candidates, budgets, seeds, scoring weights, source hashes.
- `training/<configuration>/seedN`: CSV, stdout/stderr, W&B ID, recovery checkpoint,
  immutable comparison snapshots.
- `evaluation/<phase>`: immutable spec, per-game results, per-seed CSV and ranking.
- `final_decision.json`, `REPORT.md`, `latency/`, `final_10000/`.
- `gpu_telemetry.csv`: device-wide telemetry sampled every 30 seconds.

Create `experiment_v1/STOP` to request a safe pause. Training stops at a completed
iteration and saves; evaluation finishes its current job. Do not kill all Python
processes. Remove the request only when explicitly resuming. Relaunching the manager
skips completed work. A file lock prevents duplicate managers. Source changes during
the experiment fail closed. Ordinary non-numerical job failures may retry once
from the last atomic checkpoint; repeated failure prevents subsequent stages and
final training. Numerical failures and failed preflight gates are terminal and
are never retried. The manager checks experiment-root `INTEGRITY_HOLD.json` and
`INTEGRITY_FAILURE.json` before running, and child `INTEGRITY_FAILURE.json` markers
in job logs and output directories before launch and after exit, even on exit 0.
The manager records a terminal integrity failure and failed status instead of
moving to another candidate. Partial evaluation results, missing games, mismatched
specifications or non-finite ranking inputs cannot produce a selection ranking.

CPU-only orchestration checks (mocked jobs; no CUDA imports or training):

```powershell
python -B -m unittest cuda_fdm.tests.architecture_search_manager_val -v
```

Search recovery checkpoints include physical/reconstruction state, opponent hidden
states, normalizer, optimizer, RNG states and episode state to preserve continuation.
The original preserved GRU100 is evaluation-only and is never resumed for final training.

## Integrity-fix validation update (2026-08-31)

The submission observation contract, finite-trajectory validation, episode-fixed
opponent identities/result attribution, CUDA stream ownership, and RL precision
guards have been corrected. Bounded validation passed: 53 CPU tests, actual CPU
submission adapters, GPU fault injection, live-pool save/resume and exploiter
restoration, and production-size MLP/GRU two-iteration kernel-path equivalence.
See `runs/architecture_search/experiment_v1/integrity_fixes_v1/REPORT.md` and
`fix_status.json` in that directory for evidence and limitations.

This does **not** resume or rebaseline `experiment_v1`. Its original checkpoint
remains at iteration 1450, and its `STOP`, integrity hold, manifest and provisional
results remain in place. Episode-fixed opponents are a corrected training protocol
(`cuda_episode_opponent_v2`), not an implementation-equivalent continuation of the
old pool behavior. A reviewed, distinct comparison cohort and restart approval are
required before any search or final-main training proceeds. Do not silently mix
the historical and corrected-protocol results. FP32 RL remains explicitly disabled;
the precision fix rejects an unsafe ABI rather than adding FP32 RL support.

A subsequent user-requested 200-second value audit reproduced an additional issue:
the learner adds `gamma * V(terminal_obs)` at a true match timeout, even though the
match objective ends there. Raw timeout rewards and GAE episode-boundary masks
passed, but the finite-horizon target did not. Strict `time > 200` also permits a
2001st control step. The subsequent v3 correction passed 60 CPU tests, actual
CUDA 2000-step/terminal-target tests and MLP/GRU two-iteration integration checks.
See `timeout_fixes_v1/REPORT.md` under the held experiment. Those results precede
the user-requested MLP-only updater change above. They do not release the hold.
