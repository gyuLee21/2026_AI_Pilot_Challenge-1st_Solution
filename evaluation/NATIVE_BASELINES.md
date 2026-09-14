# CPU baseline comparison

`native_baselines.py` evaluates the eight frozen Gylee/Junhwa actors against:

- vertical_deck_dive_ver01 (native BT DLL, 60 Hz)
- Release_MPC team share v7 (native MPC predictor, planning at 10 Hz)
- cutoff unreal_bt_client.exe (local UDP adapter, action-repeat 6 / 10 Hz)

The CPU FDM runs at 60 Hz; RL inference is deterministic at 10 Hz with the
current 214D observation reconstruction, saved normalization and recurrent
state. Each matchup consists of 25 seeded initial conditions with both slot
assignments: 50 games, 200-second maximum, hard deck 304.8 m, three-nine only.
All eight actors see the same opponents and initial-condition seed bank.
This uses the CPU simulator; keep its results labeled separately from GPU games.

The isolated runtime and baseline packages live under artifacts. Every process
owns its native policy state; a fresh native tree is constructed at each reset.
Initial-state swap equality is checked. Native calls, MPC fallback counts and
distinct commands are recorded; excessive fallback aborts the matchup.

Run from the repo root:

```powershell
python evaluation/native_baselines.py --spec configs/evaluation/junhwa_cross_family.local.json --output artifacts/evaluation/native_baselines_3-9_20260911 --workers 4
```

`--smoke` runs two 2-second episodes per matchup solely for interface checks.
All 24 matchups passed this smoke test (48 episodes; zero fallback calls).
Long-duration combat outcomes are collected by the full run, not inferred from smoke.

Each matchup writes progress.json after each game and result.json at completion.
The parent writes progress.json and final report.json. Rerunning the same command
reuses completed matchups only; an interrupted matchup restarts all 50 games.
Model/code/baseline hashes are saved in manifest.json; changed manifests cannot
be mixed. No BT/MPC/cutoff-vs-each-other or same-family RL matches are scheduled.
