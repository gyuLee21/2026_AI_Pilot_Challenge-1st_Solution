# Portfolio cleanup — 2026-09-14

## Scope

Private repository; no visibility change. Runtime import paths, PPO/physics/
reward/pool behavior, model weights and final release ZIPs remain unchanged.
Previously uncommitted evaluation/submission work is versioned, not discarded.

## Changes

- Source-only Release MPC under `controllers/mpc/release`.
- Custom BT node overlay under `controllers/bt/aip_dcs/BehaviorTree/BT_Content`.
- README, source provenance, contribution and runtime/artifact boundaries documented.
- Historical campaign scripts labeled rather than silently promoted to reusable APIs.
- Root `work_gpu_tables.txt` and `work_print_gpu_tables.py` moved to ignored
  `artifacts/maintenance/portfolio_cleanup_20260914/`; recoverable, not permanently deleted.
- Binary/build/archive/cache patterns excluded from Git. Required generated F-16
  source headers remain included; generated DLLs do not.

## Checks performed

| Check | Result |
|---|---|
| Source-copy SHA256 | 32 MPC and 31 BT files matched before formatting; only trailing whitespace/extra EOF blank lines normalized afterward |
| Storage/evaluation/submission/fork tests | 44 pytest tests passed |
| League/PPO CPU regression suite | 111 passed; no CUDA initialization |
| MPC predictor | MSVC x64 Release build succeeded; existing `getenv` deprecation warning |
| MPC regression suite | 10 passed against the newly built DLL |
| Tracked payload inspection | No PT/PTH/DLL/EXE/ZIP/OBJ/LIB payloads staged |
| Whitespace | `git diff --check` passed |

Total distinct tests above: **165**. These checks do not imply BT compilation,
full clean-machine SDK installation, GPU trajectory equivalence or new performance benchmarks.
Native evaluation tests use existing local SDK assets; model-based submission
tests use existing local weights. Neither is distributed with the repository.

## Final submission preservation

SHA256 values rechecked after source cleanup:

- Main 36.5k ZIP: `bf27d1eea1b45f010c6ba737b75c2842043a1f1626cbbf0912ecc6e3f43a54ac`
- Head-on 46.5k ZIP: `2afd61b9f1180130921b042a02f809954006960719fbc1b81db80d69591201f5`

These match the previously verified releases. Packaging source includes a historical
46k/60Hz Head-on entrypoint; consult `submission/README.md` before rebuilding.
