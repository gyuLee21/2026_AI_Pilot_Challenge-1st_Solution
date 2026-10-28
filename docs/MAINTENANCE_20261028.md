# Portfolio and final-lineage restoration — 2026-10-28

## Changes

- Worked from remote main `3022fcf` in a separate Git worktree. The original
  working copy's five modified/untracked CPU-recovery files were not staged,
  reverted, relocated or deleted.
- Read the latest located paper draft v2.25 and compared it with the preserved
  final 36,500 runtime/checkpoint. Restored a separate 184D entrypoint, matched
  source dependencies and actual schedule/settings, rather than replacing the
  existing 214D runtime. See `PAPER_CODE_AUDIT.md` for discrepancies left explicit.
- Archived 18 competition SDK files (9,030,925 bytes), including exactly three
  DLLs. Their byte hashes match the recorded original sources; all three DLL
  Git blob IDs also match the inspected idearendil commit. Added provenance and
  a notice that no separate redistribution license was located.
- Added explicit, idempotent runtime staging with hash verification, traversal
  checks and refusal to overwrite different existing files. No DLL is loaded
  by staging. Kept the general binary/model/log ignores; only three SDK paths
  are exceptions. Git attributes preserve archived source bytes across OSes.
- Shortened README and linked architecture, method audit, setup and source
  provenance. Kept MPC/BT source under their existing controller directories.
  Added source-only CI, dependency ranges and portable example settings.

## Validation

| Check | Observed result |
|---|---|
| Runtime bundle tests | 6 passed: hashes, corruption, traversal, duplicate paths, conflict preservation, idempotency |
| Restored lineage / launcher tests | 4 passed |
| Existing managed training tests | 4 passed |
| Existing storage policy tests | 2 passed |
| Preserved CPU suites: future_aux, main_schedule, pool_episode, pool_assigned, active_league | 80 passed |
| Preserved training parser `--help` | Imported successfully with isolated runtime and shared src paths |
| Real SDK staging, twice | 18 files verified on both passes; no DLL execution |
| New documentation links | All local targets resolved |
| Binary boundary | Only the three SDK DLL exceptions are trackable; arbitrary DLL/EXE and model artifacts remain ignored |

The 80-test suite uses toy environments, not a combat performance experiment.
Its expected empty-episode diagnostic NaNs are fixture output, not a new failed
training run. A PyTorch warning about tensor-to-scalar conversion is inherited
from an existing auxiliary-loss assertion; tests pass without changing source.

Full GPU training, CPU combat integration, clean-PC CRT provisioning, BT
compilation and submission rebuilding were not performed. No new performance
claim follows from source import or these tests. Existing weights, completed
run data, final submission ZIPs and repository visibility were not changed.
