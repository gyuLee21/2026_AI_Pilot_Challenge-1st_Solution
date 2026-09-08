# Future-position auxiliary learning

## Contract

Reference: the auxiliary task in fetched `origin/main2` (`298ca84`). Its extra
observations, critic-only opponent actions, different rewards/discounts and
altitude-hunting exploiter are **not** imported. The local 184D MLP, submission
contract, 200s finite-horizon targets, episode-fixed opponents and assigned-row
inference fixes remain in place. No remote branch was merged wholesale.

Each independent actor/critic MLP has one extra Linear head from its existing
last hidden layer to six scalars:

| Output | Lookahead | Target |
|---|---|---|
| 0:3 | opponent, 5 policy steps / 0.5s | position minus constant-world-velocity baseline |
| 3:6 | own aircraft, 10 policy steps / 1s | same residual |

For aircraft j and horizon H:

`target = R_ned_to_own_body(t) @ [p_j(t+H) - p_j(t) - v_j_ned(t)*H*0.1] / 100m`

Both residuals use the **current own aircraft body frame**, not the opponent's
body frame and not a future frame. Body velocities are converted to world/NED
before extrapolation. Pure constant-world-velocity motion has zero residual.
These are training labels, not additional observations. No future information is
available to the policy. The task does not predict velocity/acceleration or append
the future position to the four flight controls.

The actor still produces 4 independent categorical controls (21 bins = 84 logits);
the critic still produces one value. Auxiliary predictions share their respective
trunk; the actor and critic do not share parameters with each other.

## Loss and boundaries

- Actor: existing clipped PPO loss minus entropy bonus, **plus 0.1 masked MSE**.
- Critic: existing value-loss coefficient, **plus 0.1 masked MSE**.
- Average over valid xyz coordinates from both horizons, as in Git. Do not average
  the two horizons separately with equal weight when valid counts differ.
- Labels are detached. A reset anywhere in `(t, t+H]` invalidates that target,
  including the 200-second task timeout. A reset at t starts a valid new segment.
- Futures outside this rollout are masked. Do not carry labels from another
  iteration/exploiter or bridge reset states. PPO still uses every transition.
- Auxiliary predictions/physics must be finite even if their target is masked.
  Invalid raw physical trajectories abort before PPO; no `nan_to_num` workaround.

This is intentionally a changed learning objective. Identical input action
outputs before updating do not imply identical weights after adding auxiliary
gradients. It may improve the shared representation, but no short smoke test can
establish improved win rate. In particular the own-aircraft target depends on
subsequent stochastic actions; the head learns a conditional prediction under
the training policies, not an action-conditioned simulator or guaranteed planner.

## Implementation choices relative to Git

The target definition, horizons, cubic flight reward (unchanged), loss coefficient
and independent actor/critic heads are retained. Two avoidable costs are removed:

1. The existing observation kernel already computes main/opponent NED position,
   world velocity and main attitude rotation. It optionally stores those 21
   numbers per environment instead of calling `state9()` again each policy step.
   No new CUDA kernel or device-wide synchronization is added.
2. Two batched horizon operations and cumulative episode IDs build the labels;
   there is no Python loop over rollout time for padding or label generation.

Physical features stay FP64 until residual calculation, avoiding early FP32
absolute-position subtraction. Final labels/head outputs are FP32. Tests compare
the same valid targets against both the Git FP32 capture and an FP64 reference.
Masked entries are explicitly zero locally; their values do not enter MSE.

The PPO and auxiliary heads reuse a single trunk forward per network/minibatch.
Rollout, frozen-opponent inference and evaluation never calculate the auxiliary
outputs. Head initialization preserves the base policy's initial weights and RNG
state so ON/OFF starts can be compared without an extra initialization confound.

At width 512 the two heads add 6,156 parameters. At width 672 they add 8,076.
For 4096 environments and T=64, capture/target/mask buffers occupy about 48.5MiB
(21 doubles + 6 floats + 2 bools per transition). Temporary label tensors and
head activations also use memory; see measured validation results for overhead.

## CLI, checkpoint and deployment

New CLI training defaults to `--aux-pred --aux-coef 0.1`. Library callers retain
an explicit opt-in `PPOGPUConfig(aux_pred=True)` for unmodified old fixtures.
Auxiliary training requires the 184D MLP and 10Hz / substeps 6.

```powershell
# Bounded local integration check; NOT an architecture performance comparison.
python -B -m cuda_fdm.train_gpu --aux-pred --no-wandb --iters 2 `
  --milestone-period 0 --exploiter-iters 0 --sched-period 0 `
  --save runs/new_aux_smoke/checkpoint.pt --log runs/new_aux_smoke/metrics.csv

# Export keeps policy/critic only; no auxiliary modules run in submission.
python -B -m cuda_fdm.gpu_ckpt_to_bundle --ckpt runs/new_aux_smoke/checkpoint.pt `
  --output-dir runs/new_aux_smoke/bundle
```

- ON: `cuda_mlp_flat_finite_horizon_future_aux_altmix_v6` with explicit target contract.
- OFF: `cuda_mlp_flat_finite_horizon_altmix_v5` (`--no-aux-pred`).

The 2026-08-31 altitude-reward revision changes the training protocol, not auxiliary
targets or coefficient. Old v4/v5 results remain preserved, not silently resumed.
- Resume requires matching ON/OFF, auxiliary contract and loss coefficient.
  Do not silently migrate old checkpoints or append an incompatible CSV schema.
- The GPU resume regression also caught a pre-existing Adam placement issue:
  whole-checkpoint CUDA loading moves its CPU step counters to GPU. Resume now
  restores those counters to CPU for non-capturable/non-fused Adam, leaving all
  values unchanged. This avoids scalar GPU reads and preserves fresh-run layout.
- Head keys must exactly match checkpoint metadata. Bundle conversion strips
  exactly those four weight/bias tensors, then uses strict CPU model loading.
- The original architecture manager remains held and explicitly auxiliary-OFF;
  it cannot silently inherit this new default. A corrected cohort must declare
  the selected loss in its new manifest before any width/depth search resumes.
- Validation/search uses local CSV/JSON/checkpoints, no W&B. Only a future main
  run may use the previously approved W&B account; no checkpoint upload added.

## Diagnostics

Main CSV and W&B scalar logs (when enabled) record actor/critic auxiliary MSE,
opponent/own valid-target fractions, and per-horizon coordinate RMSE in metres.
The constant-velocity baseline predicts zero residual and is logged for context.
Training MSE is accumulated over the visited minibatches; it is not a held-out
prediction score or evidence of combat performance. Existing PPO metrics remain.
Exploiter metric callbacks include the same auxiliary diagnostics.

Validation entry points:

```powershell
python -B -m cuda_fdm.tests.integrity_cpu_suite --output runs/new_aux_check/cpu.json
python -B -m cuda_fdm.tests.future_aux_gpu_val --output runs/new_aux_check/gpu
```

The GPU test uses the frozen pre-change observation kernel, forced deaths and
exact 200s terminals, current-stream capture, paired 4096x64 rollouts, identical
initial PPO/Adam/RNG states for overhead timing, and exact GPU save/resume.
Reported CUDA Event durations include stream idle gaps, not pure kernel time.
