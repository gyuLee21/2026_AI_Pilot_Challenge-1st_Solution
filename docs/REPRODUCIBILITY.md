# Setup and reproducibility

## Runtime layers

1. **Source-only checks:** Python3.11 and Git. Hash/launcher/storage tests use
   the standard library, no model files, GPU context or native DLL loading.
2. **CPU policy and league tests:** the shared `requirements.txt` dependencies.
3. **CUDA training:** NVIDIA driver, a compatible CUDA-enabled PyTorch build,
   NVRTC and enough GPU memory. Choose the torch wheel for the actual system;
   `requirements.txt` is a dependency range, not a historical environment lock.
4. **Native CPU combat:** Windows x64, the provided DLLs and required CRT,
   aircraft/engine assets, models, baseline rules and the matching evaluator.
5. **MPC native build / submission build:** separate toolchains described in
   `controllers/mpc/README.md` and `submission/README.md`.

## Prepare a clean clone

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/prepare_runtime.py
python scripts/prepare_runtime.py --install
```

The second preparation command stages an isolated runtime, never replaces a
different existing DLL. Neither command tests DLL execution or installs CRTs.
Do not copy debug runtimes from unrelated installations to hide a missing
dependency. Review the SDK provider's environment requirements.

Default staging target: `artifacts/runtimes/competition`. The older evaluator's
`cpu_baselines` directory contains project-modified wrappers/providers and is
not silently replaced with the provided originals. Its additional policy files
remain local. Clone-and-stage alone does not reconstruct that full campaign.

## Select a training lineage explicitly

For the restored 184D Main source, use `--runtime competition36500 --scenario
three_nine`. Existing callers without `--runtime` continue to select the 214D
root runtime. The launcher uses a separate interpreter and working directory,
so identically named `cuda_fdm` modules from two lineages cannot share a process.
Use `--dry-run` before any real experiment. No process starts automatically.

The checked-in example and final checkpoint receipt describe different things:
an explicit fresh-run schedule versus settings at the end of a completed run.
Only the original checkpoint and its complete archive/runtime history can
support exact continuation. Results from the paper's post-competition ablation
must remain separate from competition-policy training.

## Evaluation

Supply trusted model paths in a local evaluation spec and use a new output
directory. `scripts/evaluate.py` invokes the existing GPU tournament; it does
not automatically select the restored learner runtime. CPU and GPU evaluation,
10/60Hz decision rates and initial-state/termination contracts require separate
verification; identical weights are not evidence of identical environments.

Some original integration tests require local weights, external policy packages
or build tools. Do not interpret their absence in a clean clone as a passed
integration test. See `docs/PAPER_CODE_AUDIT.md` for remaining paper/evidence
discrepancies and `docs/MAINTENANCE_20261028.md` for checks performed in this edit.
