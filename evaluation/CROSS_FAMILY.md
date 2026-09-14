# Junhwa / Gylee cross-family 3-9 evaluation

Spec: `configs/evaluation/junhwa_cross_family.local.json`.
24 pairs, 50 games per pair (25 mirrored initial-condition blocks), 1200 scored games.
No Gylee-vs-Gylee or Junhwa-vs-Junhwa games. All eight face submission4499.

`junhwa_actor` uses the frozen actor and saved observation normalization, with
per-lane GRUCell state reset on episode start. Shared observation semantics are
assumed from common code provenance, not proved by tensor dimensions alone.

`start_after_training.ps1` waits for training exit; `after_training.py` requires
the clean +2000 receipt/status and internal iteration72000 in final/snapshot
checkpoints. It runs 16 unscored GPU smoke games, then the scored evaluation.
Failures stop the pipeline and are recorded in launch_status.json.

Output: pairs/*.json, progress.json, gpu_preflight.json, report.json,
rankings.csv, score_matrix.csv. Prefer report.rankings_by_family for comparisons;
the combined list has different opponent sets. STOP halts after a complete pair.
Automatic launcher is one-shot. An intentional restart needs review of launch.lock;
completed pair records can be resumed with the unchanged manifest/code.
