"""Read-only actor-trunk gradient diagnostic; does not tune or step an optimizer."""
import argparse
import gc
from pathlib import Path

import torch

from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.future_aux import auxiliary_error
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.search_eval import atomic_json


def measure(trainer, minibatches=8):
    adv, _, _ = trainer.collect_rollout()
    obs, acts = trainer.b_obs.flatten(0, 1), trainer.b_act.flatten(0, 1)
    labels, masks = trainer.b_aux_labels.flatten(0, 1), trainer.b_aux_mask.flatten(0, 1)
    old_logp, advantages = trainer.b_logp.flatten(), adv.flatten()
    params = [p for layer in trainer.model._actor_feature_layers for p in layer.parameters()]
    generator = torch.Generator(device=obs.device).manual_seed(82473)
    indices = torch.randperm(len(obs), device=obs.device, generator=generator)
    records = []
    for mb in indices.tensor_split(minibatches):
        logp, entropy, prediction = trainer.model.evaluate_actions_with_aux(obs[mb], acts[mb])
        a = advantages[mb]
        a = (a - a.mean()) / (a.std() + 1e-8)
        ratio = (logp - old_logp[mb]).exp()
        policy = torch.maximum(-a * ratio, -a * ratio.clamp(.8, 1.2)).mean()
        task = policy - trainer.cfg.ent_coef * entropy.mean()
        aux, _, _ = auxiliary_error(prediction, labels[mb], masks[mb])
        gp = torch.autograd.grad(task, params, retain_graph=True)
        ga = torch.autograd.grad(trainer.cfg.aux_coef * aux, params)
        p2 = sum(g.square().sum() for g in gp)
        a2 = sum(g.square().sum() for g in ga)
        dot = sum((p * a).sum() for p, a in zip(gp, ga))
        values = torch.stack((policy.detach(), aux.detach(), p2.sqrt(), a2.sqrt(),
                              (a2 / p2.clamp_min(1e-30)).sqrt(),
                              dot / (p2 * a2).sqrt().clamp_min(1e-30)))
        require_finite(values, "actor shared-trunk gradient balance")
        records.append(dict(zip(("policy_loss", "aux_mse", "ppo_entropy_grad_norm",
                                 "weighted_aux_grad_norm", "aux_to_ppo_ratio", "cosine"),
                                values.cpu().tolist())))
    return records


def run(checkpoint, output):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = PPOGPUConfig(**saved["cfg"])
    if not cfg.aux_pred or cfg.aux_coef != .1 or cfg.sched_period != 0:
        raise ValueError("diagnostic expects the frozen auxiliary-ON coefficient-0.1 smoke")
    nenv = len(saved["runtime"]["trainer"]["opp_assign"])
    reports = {}
    for tag in ("initial", "after_two_updates"):
        env = GpuDogfightVecEnv(nenv, substeps=6, seed=cfg.seed)
        trainer = PPOGPUTrainer(env, cfg)
        if tag == "after_two_updates":
            trainer.load(checkpoint)
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        rows = measure(trainer)
        if not all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items()):
            raise AssertionError("gradient diagnostic unexpectedly changed parameters")
        ratios = [r["aux_to_ppo_ratio"] for r in rows]
        reports[tag] = dict(minibatches=rows, mean_ratio=sum(ratios)/len(ratios),
                            max_ratio=max(ratios), min_ratio=min(ratios))
        del trainer, env, before
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(passed=True, coefficient=.1, checkpoint=str(checkpoint),
                  optimizer_steps=0, measure="pre-clipping shared actor-trunk gradients; PPO includes entropy",
                  caveat="Initial/2-update observations only, not a coefficient optimum or long-run guarantee",
                  results=reports)
    atomic_json(output, result)
    print({tag: {k: v for k, v in row.items() if k != "minibatches"}
           for tag, row in reports.items()}, flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        run(args.checkpoint, args.output)
    except Exception as exc:
        record_integrity_failure(args.output.parent, exc, "aux_gradient_balance")
        raise
