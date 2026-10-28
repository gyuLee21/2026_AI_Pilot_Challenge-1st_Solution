# Paper / source / final-checkpoint audit — 2026-10-28

## Scope

Reference: `AI_Pilot_Paper_Draft_v2_25_평가상대표_삭제_노란색_수정표시.docx`,
Drive file ID `19i_wxdIHuOtpD2vQqtMNP2Hj11bQXEXf` (2026-10-09).
The manuscript is read-only; it is not redistributed in this repository.
Compared against GitHub main `3022fcf` and the preserved 2026-09-14 final
36,500-iteration runtime and checkpoint. Weights and submission ZIPs are unchanged.

The root `cuda_fdm` is a different, 214D lineage. Replacing individual modules
with the 184D final Main's modules would mix observation/kernel/checkpoint
contracts. The final lineage is therefore restored under
[`research/competition_36500`](../research/competition_36500/README.md), selected
explicitly with `scripts/train.py --runtime competition36500`. The default
launcher continues to use the root runtime; historical users are not migrated
silently. Hash receipts identify the restored source, not a claim of retraining.

## Restored, evidence-backed implementation

| Item | Root before restoration | Actual final 3-9 lineage / restored path |
|---|---|---|
| Actor / critic input | 214D lineage | Both 184D, independent MLP 384×3 |
| Initial aircraft fuel | 6,000 lb | 3,000 lb = 1,500 + 1,500 |
| FCS component step | Historical 1/120 s | Adaptation branch 1/60 s |
| First physical state | No final neutral-tick initializer | Exactly one neutral physical tick, including evaluation and cached autoresets |
| Archive / pool | Separate vNext adapter implementation | Original `LeagueArchive` and active24 path, kept with its matching trainer |
| Payoff evaluation | Older per-pair path | Preserved batched row completion plus its matching parallel evaluator |
| External actors | Frozen 214D opponents absent in this lineage's public source | `external_opponents.py` and observation kernel restored; not learner initializers |
| Final settings | CLI defaults / unrelated run settings | Recorded checkpoint settings and bounded schedule supplied separately |

The active layout is 1 latest, ≤4 recent, ≤16 core, ≤3 challengers. Frozen
opponents and all archived policies still need separately retained payloads.
Restoring source does not put old opponents or original weights in a clone.

## Manuscript statements that are not historical facts

These are **not** repaired by inventing a new implementation and attributing
old scores to it. A future paper-aligned experiment would require a new
protocol, fresh training/evaluation and separate results.

| Manuscript | Preserved evidence | Treatment |
|---|---|---|
| Privileged critic receives opponent action history | Both final input-layer tensors are `(384,184)`; same observation input | Do not add 20 dimensions retroactively. Manuscript correction remains necessary |
| LR fixed at 1e-4; rollout64 for the whole final run | Bounded schedule; final LR5e-5, rollout104, entropy1e-4, gamma0.997 | Publish final settings and schedule, not a fictitious constant history |
| Geometry reward is solely the stated potential difference | Preserved final reward uses the final-safe geometry occupancy/trapezoid implementation with budget ±5; shaping decays to zero | Preserve the actual reward source; no post-hoc objective replacement |
| No terminal kill/timeout rewards | Preserved source includes kill ±5 and timeout +5/-5/draw-4 | Flag text/source mismatch; do not erase the rewards that produced the weights |
| Every exploiter stops at 0.8 / two confirmations | Final checkpoint config records 0.75; thresholds/roles changed across campaign history | Expose config/history limits; do not label one current setting as all-time behavior |
| Recovery UCB95 ≤0.65 | Restored final league uses 0.70 | Preserve the actual implemented gate; a change to 0.65 is a new protocol |
| All 11 final opponents were absent from the training pool | Preserved intervention history includes 45k/67k and Survive entries | This is a manuscript/history question, not a reason to delete archive records |

The final actor's own action history (20 values in the 184D input), the future
position auxiliary heads, and the separate 30D acceleration expansion of the
214D lineage are distinct mechanisms. None implies a 20D privileged critic.

## Reproduction boundary

The example launcher is safe to inspect with `--dry-run`. A fresh run starting
with corrected physics is not an exact replay of the older physical conditions,
manual opponent interventions and reward edits. Exact continuation needs the
trusted original checkpoint, full archive, RNG/runtime state and original paths
or a separately validated migration. Do not execute teammate pickle/PT files
merely to inspect them.

The paper's post-competition NoExp/StdExp/DivExp experiment is a separate cohort,
not the final submission lineage. Its settings/results must not be used to
rewrite final-checkpoint history. No new training, tournament, W&B run or model
conversion is launched during this restoration.
