# Local artifacts (not published)

Only this README is tracked. Model payloads, external source, local manifests
and results underneath this directory are ignored by Git.

- `models/rl/3-9/`: 3-9 checkpoints
- `models/rl/headon/`: Head-on checkpoints
- `models/rl/common/`: shared/mixed-training models, original20k and 4499
- `models/bt/`: named DLL/XML packages
- `models/mpc/`: named MPC packages (dependency validation required)
- `models/external/inbox/`: unreviewed teammate packages
- `models/external/approved/`: reviewed packages, still explicitly selected
- `models/catalog.local.json`: local verified import provenance
- `evaluations/`: recommended location for future evaluation outputs

See [storage and execution guide](../docs/STRUCTURE.md). Cloning the repository
does not download model payloads. The training launcher creates its run directory
on execution; create baseline/external package folders when importing a package.
