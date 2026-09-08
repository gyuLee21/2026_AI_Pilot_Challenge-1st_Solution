"""Training-only short-horizon position residuals; never policy observations.

Feature layout (FP64, main perspective): main NED pos/vel, opponent NED pos/vel,
main NED-to-body rotation (row-major). Labels are opponent +5 steps and main
+10 steps, minus a constant-world-velocity prediction, in the current main body
frame, divided by 100 metres. Cross-episode and unavailable futures are masked.
"""
import math

import torch

AUX_FEATURE_DIM = 21
AUX_DIM = 6
AUX_OPP_H = 5
AUX_SELF_H = 10
AUX_POS_SCALE_M = 100.0
AUX_PROTOCOL = "cuda_mlp_flat_finite_horizon_future_aux_diverse_h3_reset_v8"
AUX_CONTRACT = "main_body_cv_residual_opp5_self10_scale100_v1"
AUX_STATE_KEYS = frozenset(("actor_aux_head.weight", "actor_aux_head.bias",
                            "critic_aux_head.weight", "critic_aux_head.bias"))
AUX_METRIC_KEYS = ("aux_actor_mse", "aux_critic_mse", "aux_baseline_mse",
                   "aux_opp_coverage", "aux_self_coverage",
                   "aux_actor_opp_rmse_m", "aux_actor_self_rmse_m",
                   "aux_critic_opp_rmse_m", "aux_critic_self_rmse_m",
                   "aux_baseline_opp_rmse_m", "aux_baseline_self_rmse_m")


@torch.no_grad()
def build_future_labels(features, episode_starts, dt):
    """Two vectorized horizon operations; no Python loop over rollout steps.

The PPO batch is NOT shortened or repacked. Invalid auxiliary targets are zero
with a false mask, while the same transitions still participate fully in PPO.
    """
    if features.ndim != 3 or features.shape[-1] != AUX_FEATURE_DIM:
        raise ValueError("auxiliary features must have shape (T, environments, 21)")
    if episode_starts.shape != features.shape[:2] or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid auxiliary episode shape or control interval")
    time, environments = features.shape[:2]
    labels = torch.zeros(time, environments, AUX_DIM, device=features.device, dtype=torch.float32)
    mask = torch.zeros(time, environments, 2, device=features.device, dtype=torch.bool)
    episode_ids = episode_starts.to(torch.int64).cumsum(0)
    for slot, horizon, pos, vel in ((0, AUX_OPP_H, 6, 9), (1, AUX_SELF_H, 0, 3)):
        if time <= horizon:
            continue
        current = features[:-horizon]
        residual = (features[horizon:, :, pos:pos + 3] - current[:, :, pos:pos + 3]
                    - current[:, :, vel:vel + 3] * (horizon * dt))
        rotation = current[:, :, 12:21].reshape(time - horizon, environments, 3, 3)
        target = torch.matmul(rotation, residual.unsqueeze(-1)).squeeze(-1) / AUX_POS_SCALE_M
        valid = episode_ids[:-horizon] == episode_ids[horizon:]
        labels[:-horizon, :, 3 * slot:3 * slot + 3] = torch.where(
            valid.unsqueeze(-1), target, torch.zeros_like(target)).float()
        mask[:-horizon, :, slot] = valid
    return labels, mask


def auxiliary_error(prediction, target, mask):
    """Masked MSE plus per-horizon SSE/counts for device-side logging.

No nan_to_num: non-finite predictions must poison loss/gradient checks rather
than silently passing because the corresponding label happens to be masked.
    """
    squared = (prediction - target).reshape(-1, 2, 3).square().sum(-1)
    sums = (squared * mask).sum(0)
    counts = mask.sum(0).to(sums.dtype) * 3.0
    return sums.sum() / counts.sum().clamp_min(1.0), sums, counts


def inference_state_dict(state_dict, enabled=False):
    """Strip exactly the four declared training-only tensors, reject mismatches."""
    present = {key for key in state_dict if key.startswith(("actor_aux_head.", "critic_aux_head."))}
    if present != (AUX_STATE_KEYS if enabled else set()):
        raise ValueError("checkpoint auxiliary metadata and prediction-head keys disagree")
    return {key: value for key, value in state_dict.items() if key not in AUX_STATE_KEYS}
