# Portfolio cleanup implementation plan

> Execute inline; the user explicitly requested no subagents.

**Goal:** Maintain a private, source-focused portfolio repository for RL, MPC and BT.

**Architecture:** Keep existing runtime import paths. Add controller source snapshots under `controllers/`; keep model payloads, SDK binaries and run output under ignored `artifacts/`. Preserve original source attribution.

**Tech stack:** Python/PyTorch/CUDA, C++17, PowerShell, Git.

**Spec:** User-approved structure in the 2026-09-14 conversation.

## Constraints

- Do not change repository visibility or rewrite published history.
- Do not modify final submission ZIPs, policy weights or original training directories.
- Do not change PPO, reward, physics, pool or inference behavior for cleanup.
- Keep existing uncommitted functional work, with accurate commit descriptions.
- Back up disposable local files before removing them from the source tree.

## Tasks

- [x] Establish baseline: run storage and evaluation unit tests; inspect Git changes.
- [x] Import Release MPC source/config/tests without binaries; preserve directory-relative paths.
- [x] Import BT custom-node source overlay without copying the entire third-party SDK.
- [x] Remove unused root scratch files to ignored recovery storage; strengthen binary/output ignores.
- [x] Rewrite README and structure/provenance documentation; distinguish runtime code from historical campaigns.
- [x] Verify source copies by SHA256, run regression tests and MPC source tests, review staged files and secret patterns.
- [x] Commit existing evaluation/submission work separately from controller imports and documentation.
- [ ] Non-force push the reviewed branch, then fast-forward main only if its remote history is an ancestor. Verify remote SHA and private visibility.

## Verification commands

```powershell
python -m unittest discover -s tests
python -m unittest discover -s evaluation -p 'test_*.py'
python -m cuda_fdm.tests.vnext_cpu_suite
python -m pytest controllers/mpc/release/tests -q
git diff --check
git diff --cached --stat
```

Native MPC tests require a locally built predictor DLL. CPU simulator integrations require separately obtained SDK assets; report unavailable checks rather than claiming a clean-machine build.
