# CPU reset-state audit — 2026-09-13

## Corrected locally

`artifacts/runtimes/cpu_baselines/FighterSim.py` previously returned a zero-filled
sensor vector with only requested position, attitude and health assigned. The
FDM itself had nonzero velocity. Reset now reads `get_fdm_data()` and uses the
existing `_update_state()` conversion, without calling `Run` or advancing time.
This runtime artifact is git-ignored; retain this local correction when replacing
or redistributing the CPU runtime. Training runtime and weights were not changed.

The DLL initialization already reports time 1/60 s. Thus actual initial position
differs from requested IC by a few metres. Headon checks now validate requested
distance separately and bound actual displacement by speed × initial time plus
0.5 m for exported coordinate quantization. Evaluator manifest includes FDM wrapper
hash to reject reuse across runtime changes.

## Verification

- `work/test_cpu_reset_state.py`: failed on old zero velocity; passes after fix.
  Full reset vector equals FDM conversion, repeated reset matches, no extra time step.
- Headon 5539 m: two mirrored 1-second smoke games pass.
- 3-9: same requested IC seed bank, 25 mirrored blocks, 50 games, 7 CPU workers,
  argmax 10 Hz, 200-second limit, current-lineage iteration 30000 versus 69000.

| Evaluator | W/L/D |
|---|---|
| Original CPU | 21/22/7 |
| Corrected full-state CPU | 19/24/7 |
| GPU | 7/40/3 |

27/50 outcomes differ from original CPU; 26/50 still differ from GPU. Initial
state bug is fixed, but full CPU/GPU evaluation equivalence is NOT established.
Small numerical differences can amplify through closed-loop decisions; this does
not prove the remaining difference is harmless numerical noise. GPU was matched
to requested physical ICs, not byte-identical DLL internal state at 1/60 s.

Detailed results: `artifacts/evaluations/current30k_full_reset_corrected_20260913/`.
Earlier CPU tables/heatmap are pre-correction and should not be treated as corrected
competition estimates. No broad tournament rerun was performed in this fix.

## Outstanding

Actual competition-server first packet not captured: UDP 9999 was not open.
Submission client consumes received PlaneInfo velocity; synthetic wire checks are
not evidence of real server initial state. Training remains stopped at 34924.
