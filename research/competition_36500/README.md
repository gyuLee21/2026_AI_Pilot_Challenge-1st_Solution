# Final 3-9 training lineage — restored 184D runtime

This is the preserved source used by the final 36,500-iteration Main branch,
not a new algorithm and not a freshly trained model. It restores post-20k work
missing from the public root's separate 214D/Head-on lineage without silently
changing that lineage's imports or checkpoint protocol.

## Restored behavior

- Actor and critic inputs are both **184D**, independent Tanh MLP 384×3 with
  4×21 action logits and training-only future-position auxiliary heads.
- Initial fuel is **3,000 lb per aircraft (1,500 + 1,500)**, not 6,000 lb.
- The adaptation branch uses **FCS component dt = 1/60 s** in both the CUDA
  kernel and Python reference. This is distinct from the historical 1/120 golden.
- Exactly one neutral `[0, 0, 0, 1]` physical tick initializes the first observed
  state, paired evaluation states and cached autoreset banks consistently.
- The actual lineage's `LeagueArchive`, role/profile scheduler, 70% UCB recovery
  gate, external-214D frozen opponent adapter and batched payoff evaluator remain
  together. Do not combine individual files with the root vNext implementation.
- The bounded LR/entropy/rollout schedule is retained. The final checkpoint used
  rollout104, actor/critic LR5e-5, entropy coefficient1e-4 and gamma0.997.

`source_manifest.json` gives the SHA-256 of each original source file.
`checkpoint_settings.json` contains configuration and tensor **shapes only**;
it contains no model weights, optimizer tensors, RNG state or opponent payloads.
Original comments sometimes describe earlier settings; the manifest, final
constants and [paper/code audit](../../docs/PAPER_CODE_AUDIT.md) distinguish them.

## Entry point

From the repository root, inspect the command without starting a GPU or writing
run data:

```powershell
python scripts/train.py --runtime competition36500 --scenario three_nine --run-name reproduce184 --config configs/training/competition36500.example.json --dry-run
```

Remove `--dry-run` only when intentionally starting a **new experiment**.
The launcher runs a separate interpreter from this runtime root, with the
repository's shared `src` on PYTHONPATH. Existing 214D modules are not imported
into this runtime. No training was started as part of repository maintenance.

The example specifies a corrected-physics fresh run and the bounded schedule;
it does **not** replay all reward interventions, manually inserted opponents,
or old physical conditions from the original 20k→36.5k campaign. Reproducing the
submitted weights exactly requires the original checkpoint, archive, RNG/runtime
state and intervention history. An old copied checkpoint may embed local archive
paths; do not claim arbitrary checkpoint copies are portable resumes.

## Paper versus measured history

The paper draft is used to identify intended components, not to fabricate
historical settings. Its privileged critic, potential-difference shaping and
some fixed LR/rollout/side-learner statements differ from preserved evidence.
They are documented in the audit rather than implemented retroactively and
described as if they produced the original results.
