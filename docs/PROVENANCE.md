# Source provenance and reproducibility boundary

This repository is private. Cleanup does not authorize public redistribution.

| Component | Origin / treatment |
|---|---|
| `cuda_fdm`, `claude_code`, `src/dogfight` | Existing project history; retains the inherited simulator, policy and protocol code. Not represented as entirely original work. |
| `controllers/mpc/release` | Source/config/tests copied from local `Release_MPC_team_share`. Original directory-relative imports retained. Binary/build products excluded. |
| MPC aircraft/engine XML and generated F-16 data | Inherited model data; preserve headers and source attribution. Check upstream terms before publishing. |
| `controllers/bt/aip_dcs/BehaviorTree/BT_Content` | Source overlay from local AIP_DCS. Requires the original SDK; does not vendor the entire BehaviorTree.CPP framework. |
| `evaluation/junhwa_policy.py` | Project adapter for external policy formats; teammate weights/packages are not distributed. |
| `artifacts` | Local-only data boundary. Original runs, teammates' packages, SDKs, checkpoints and releases are not silently downloaded. |

The original desktop MPC/BT projects and original run folders remain untouched.
SHA256 comparisons verified the initial copies. Trailing whitespace/extra EOF blank
lines were then normalized without changing source tokens; import checks do not establish authorship or license clearance.
No new repository-wide license is applied. Existing copyright notices remain in source.

## What a clone can and cannot reproduce

Source-only tests cover storage selection, ranking/resume contracts and league state logic.
Full training additionally needs the GPU runtime and explicit run configuration.
Resuming historical training needs its checkpoint and archive.
CPU match evaluation needs separately supplied SDK/DLL assets and model files.
MPC native tests need the locally built predictor. BT compilation needs the original SDK.
Final submission executables remain independently versioned release artifacts outside Git.
