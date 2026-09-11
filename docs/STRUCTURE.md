# Repository and model storage

## Code layout

```text
aipilot-rl/
  cuda_fdm/           GPU environment, PPO, league, stable training CLI
  claude_code/        checkpoint-compatible networks and observation references
  src/dogfight/       common environment contracts and geometry
  evaluation/         tournament engine and verified policy adapters
  scripts/            thin train/evaluate entrypoints
  configs/
    training/         portable PPO argument examples
    evaluation/       portable examples and ignored local campaign rosters
    models/           model registration example
  tests/              storage and launcher tests
  docs/               usage, design, contribution guidelines
  artifacts/          local-only payloads; only README is tracked
```

Existing `cuda_fdm/tests` and `evaluation/test_*.py` remain at stable paths.
Do not move the active evaluator or change imported modules mid-campaign.
`claude_code` is a compatibility package, not a second active training engine.

## Local artifact layout

```text
artifacts/models/
  rl/
    3-9/<run-name>/
    headon/<run-name>/
    common/<run-or-model-name>/
  bt/<model-name>/                     DLL + XML pair
  mpc/<model-name>/                    source/config/dependency snapshot
  external/
    inbox/<team>/<model-version>/      unreviewed teammate package
    approved/<team>/<model-version>/   reviewed package, explicit evaluation only
  catalog.local.json                  local provenance and verified hashes
```

New managed training runs store `checkpoint.pt`, retained `iter_*.pt`,
`training.csv` and `league/` under the scenario/run directory.
`three_nine` maps to directory `3-9`; `headon` maps to `headon`; training
scenario `mixed` maps to `common`. `common` is also where shared benchmark
models such as original20k and submission4499 are stored. It does not imply
that the current separate leagues use mixed initial conditions.

On 2026-09-11, verified copies of 58 RL models, six BT packages, and one
ROM NMPC snapshot were placed here. The RL payloads occupy approximately
6.31 GB (decimal). Source models, complete training checkpoints and original
archive locations remain unchanged. Local `catalog.local.json` records paths
and hashes; MPC has a tree digest over sorted `relative_path:SHA256` lines.
The MPC copy excludes archives, caches and build intermediates; external
dependencies and GPU inference are NOT validated.

Imported snapshots under `completed_70k`, `completed_20k_resume` and
`legacy20k_obs214` are evaluation storage, not freshly relocated training runs.
Their embedded archive paths may refer to the preserved original directories.
Do not resume them as portable, self-contained runs. Resume historical training
using its original paths and explicit `cuda_fdm.train_gpu` invocation.

## Managed training

Run from the repository root. This launcher changes paths only, not PPO values:

```powershell
python scripts/train.py --scenario three_nine --run-name experiment01 --dry-run -- --iters 20000 --rollout 64
python scripts/train.py --scenario headon --run-name experiment02 --config configs/training/minimal.example.json --dry-run
```

Inspect the printed command and remove `--dry-run` to actually train.
The example config is illustrative, NOT the completed competition schedule.
Forward any intended reward/LR/exploiter settings after `--` or in the config's
`arguments` array. Existing nonempty run directories require `--resume`:

```powershell
python scripts/train.py --scenario headon --run-name experiment02 --resume --config configs/training/minimal.example.json
```

Keep the same run settings when resuming. The trainer still validates checkpoint
scenario, reward and schedule contracts. Managed path overrides are refused;
the legacy CLI remains available for old runs with custom paths.

## Evaluation

```powershell
python scripts/evaluate.py --spec configs/evaluation/three_nine.local.json --output artifacts/evaluations/three_nine --validate-only
python scripts/evaluate.py --spec configs/evaluation/headon.local.json --output artifacts/evaluations/headon --validate-only
```

Remove `--validate-only` to run. Local rosters use the copied model paths and
are intentionally ignored by Git. They contain 27 scenario-specific models,
original20k 15000/17500/20000, and submission4499: 31 entrants each.
The only shared entrants between the two rosters are those four baselines.
Do not replace an existing result manifest with one using different paths.
The currently running league retains its original roster paths and output
directory outside this repository; the new configs do not change that run.

See [evaluation protocol](../evaluation/README.md) for deterministic action,
mirrored games, 5539m Head-on distance and restart semantics.

## External models and baselines

Store teammate code and weights together under `external/inbox`, not inside
`src`, `cuda_fdm` or a directory on PYTHONPATH. Record ownership, source/version,
license/sharing permission, hashes, observation/action definitions, recurrent
state/reset rules, dependencies and desired scenario in a package manifest.
Do not execute install scripts or deserialize unknown pickle/PT files merely
to inspect a package. Untrusted PyTorch/pickle files can execute code.

After review, place the package in `approved` and explicitly select a verified
adapter and evaluation roster. Moving a folder alone does not certify it.
No teammate packages have been imported yet. BT/MPC packages are stored only;
the GPU tournament currently accepts compatible checkpoints and the verified
legacy184 bundle adapter. No automatic BT/MPC/external execution exists.

## Validation

```powershell
python -m unittest discover -s tests
python -m unittest discover -s evaluation -p "test_*.py"
python -m cuda_fdm.tests.vnext_cpu_suite
```
