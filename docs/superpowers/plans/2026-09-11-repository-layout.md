# Training, Evaluation and Model Storage Implementation Plan

> For agentic workers: use superpowers:executing-plans inline; the user prohibits subagents.

**Goal:** Organize runnable code and local models without invalidating the ongoing separated GPU league.

**Architecture:** Keep existing import packages and evaluation entrypoints stable. Add thin managed launchers, configuration directories, tests, and ignored artifact directories. Copy existing model assets with checksums; never move source checkpoints or archive files. Imported teammate packages are inert data until reviewed.

**Tech Stack:** Python standard library, existing PyTorch/CUDA modules, PowerShell, Git.

**Spec:** User-approved structure in this conversation: cuda_fdm, evaluation, src, configs, scripts, tests, docs, artifacts; additional BT/MPC/external isolation and RL 3-9/headon/common storage.

## Global Constraints

- No changes to reward, optimizer, pool, active evaluation code or checkpoint contents.
- RL roots: `artifacts/models/rl/3-9`, `headon`, `common`.
- Baselines: `artifacts/models/bt`, `artifacts/models/mpc`.
- Teammates: `artifacts/models/external/inbox` and `approved`; neither is a Python import path.
- Large models, imported third-party source, secrets and local absolute-path configs stay out of Git.
- Existing legacy CLI and absolute checkpoint references remain valid.

## Task 1: Preserve the current tested source baseline

- [x] Review the current diff and record the previously verified 111 training and 5 evaluation tests.
- [x] Create an in-place feature branch unless a separate worktree is explicitly approved. Never change active evaluator paths.
- [x] Commit training snapshot and scenario-separated evaluation as separate Conventional Commits.

## Task 2: Managed training paths and launchers

Files: `scripts/train.py`, `scripts/evaluate.py`, `tests/test_managed_training.py`.

- [x] Write subprocess tests: dry-run three_nine emits `artifacts/models/rl/3-9/demo/checkpoint.pt`; headon emits `headon/demo`; mixed emits `common/demo`.
- [x] Test that `../escape`, overriding `--save`, and an existing checkpoint without `--resume` fail. Test managed resume reuses the exact checkpoint and directory.
- [x] Run tests and observe missing-launcher failure.
- [x] Implement a standard-library launcher using argparse, subprocess, pathlib; `--dry-run` prints the command and does not create directories. Forward PPO arguments without changing their values. Reject overrides to managed scenario/save/log/league paths.
- [x] Keep the original `python -m cuda_fdm.train_gpu --save ...` for historical resumes.
- [x] Add an evaluation forwarding entrypoint without changing the running tournament module.
- [x] Run tests and both CLI help commands, then commit `feat(storage): add scenario-scoped training launcher`.

## Task 3: Assets and external staging

Files: `.gitignore`, `artifacts/README.md`, `configs/models/catalog.example.json`, local ignored catalog.

- [x] Add ignored roots and portable example configuration files; no machine-specific campaign JSON in Git.
- [x] Copy BT DLL/XML pairs from the identified local baseline repositories into named BT packages; verify SHA256 before declaring success.
- [x] Copy the identified ROM NMPC package, excluding historical backups, caches and build intermediates, into its own local MPC package. Mark dependency validation and GPU adapter as not performed.
- [x] Copy saved main checkpoints by scenario and common 20k/4499 artifacts into categorized local storage, preserving originals and hashes. Do not copy the whole archive or relocate its references.
- [x] Create empty external inbox/approved directories. Do not invent teammate packages, execute their scripts, or claim GPU support.
- [x] Inspect `git status --ignored` and candidate staged files to ensure no payload or local path configuration can be pushed.

## Task 4: Documentation, final validation and publishing

Files: `README.md`, `docs/STRUCTURE.md`, `docs/CONTRIBUTING.md`, `tests/test_storage_policy.py`.

- [x] Document actual paths, commands, provenance, unsupported adapters, and feature/fix/refactor/chore/test/docs commit conventions.
- [x] Test Git ignore behavior using representative .pt, DLL, external source, and local catalog paths.
- [x] Run `python -m unittest discover -s tests`, `python -m unittest discover -s evaluation`, and `python -m cuda_fdm.tests.vnext_cpu_suite`.
- [x] Compare active league manifest/evaluator hashes, check progress, and self-review the diff without subagents.
- [x] Commit layout changes by purpose, push only to the existing `leeai021213-afk/aipilot-rl` origin, and report branch and commit hashes. Never force push.

## Completion evidence

- 2026-09-11: 111 training/league, 5 evaluation and 6 storage/launcher tests passed.
- Both categorized 31-model local evaluation manifests passed validation.
- Active evaluator, adapter, search evaluator and environment hashes were unchanged.
- Copies of 58 RL models and six BT packages passed per-file SHA256 comparison;
  the stored ROM NMPC snapshot has 416 verified files. Source paths remain intact.
- Pushed `refactor/model-storage-layout` without merging main or rewriting history.
  GitHub redirected the existing origin to `https://github.com/gyuLee21/aipilot-rl.git`.
