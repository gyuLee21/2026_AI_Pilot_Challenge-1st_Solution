"""Fixed, paired 10 Hz CUDA checkpoint evaluation. No training or pool mutation."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cuda_fdm.ppo_gpu import build_actor_critic, RunningNorm, action_to_env
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.future_aux import inference_state_dict


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_policy(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg, sd = ckpt["cfg"], ckpt["model"]
    require_finite(sd, f"checkpoint_model:{path}")
    sd = inference_state_dict(sd, enabled=bool(cfg.get("aux_pred", False)))
    architecture = cfg.get("architecture", "gru" if "actor_gru.weight_ih_l0" in sd else "mlp")
    # Infer the input width from the frozen checkpoint.  The current CUDA
    # contract is 214D; retaining a hard-coded 184 here silently breaks every
    # evaluation after the approved acceleration-feature extension.
    first_layers = [v for k, v in sd.items()
                    if torch.is_tensor(v) and v.ndim == 2 and k.endswith("weight")
                    and ("actor" in k or "critic" in k)]
    if not first_layers:
        raise ValueError(f"cannot infer observation width from checkpoint {path}")
    obs_dim = int(first_layers[0].shape[1])
    kwargs = dict(obs_dim=obs_dim, act_dim=4, num_bins=cfg.get("num_bins", 21),
                  architecture=architecture, hidden=tuple(cfg["hidden"]),
                  gru_size=cfg.get("gru_size", 256), encoder_depth=cfg.get("encoder_depth", 2),
                  activation=cfg.get("activation", "tanh"))
    model = build_actor_critic(**kwargs).to(device).eval()
    model.load_state_dict(sd, strict=True)
    for p in model.parameters():
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError(f"non-finite checkpoint {path}")
        p.requires_grad_(False)
    norm = None
    if ckpt.get("norm") is not None:
        require_finite(ckpt["norm"], f"checkpoint_normalization:{path}")
        norm = RunningNorm(obs_dim, device)
        norm.load_state_dict({k: v.to(device) for k, v in ckpt["norm"].items()})
    return model, norm


def summarize(records):
    scores = np.array([r["score"] for r in records])
    # Bootstrap paired IC blocks, not individual mirrored games.
    pairs = (scores[:len(scores)//2] + scores[len(scores)//2:]) * 0.5
    rng = np.random.default_rng(9241)
    means = pairs[rng.integers(len(pairs), size=(2000, len(pairs)))].mean(1)
    hits = [r["first_hit_sec"] for r in records if r["first_hit_sec"] is not None]
    return dict(games=len(records), score=float(scores.mean()),
                score_ci95=np.quantile(means, [.025, .975]).tolist(),
                wins=sum(r["score"] == 1 for r in records),
                draws=sum(r["score"] == .5 for r in records),
                losses=sum(r["score"] == 0 for r in records),
                damage_diff=float(np.mean([r["damage_diff"] for r in records])),
                crash_rate=float(np.mean([r["crash"] for r in records])),
                first_hit_sec=float(np.mean(hits)) if hits else None,
                first_hit_fraction=len(hits)/len(records),
                duration_sec=float(np.mean([r["duration_sec"] for r in records])))


@torch.inference_mode()
def evaluate_pair(env, a, b, scenario, seed, reset_a=False):
    """Exactly one full episode per lane; second half swaps policy roles on identical ICs."""
    n, half = env.nenv, env.nenv // 2
    torch.manual_seed(seed)
    env.rng = np.random.default_rng(seed)
    env.scenario_b_prob = float(scenario == "headon")
    env.ic_pool = None
    env.ic_pool_size = 128
    env.reset(stagger=False)
    states = env.sim.states.view(n, 2, -1)
    states[half:].copy_(states[:half])
    env.obr.reset_all()
    env.obr.kernel_init_reward_state(env.sim.states)
    obs_dim = int(getattr(env, "OBS_SIZE", 214))
    obs = env.obr.kernel_build_obs(env.sim.states).view(n, 2, obs_dim)
    slots_a = torch.cat((torch.zeros(half), torch.ones(half))).long().cuda()
    slots_b = 1 - slots_a
    rows = torch.arange(n, device="cuda")
    ma, na = a; mb, nb = b
    ha, hb = ma.initial_state(n, "cuda"), mb.initial_state(n, "cuda")
    finished = torch.zeros(n, dtype=torch.bool, device="cuda")
    starts = torch.ones(n, device="cuda")
    first_hit = torch.full((n,), -1., device="cuda")
    first_hit_b = first_hit.clone()
    records = [None] * n
    started = time.perf_counter()
    for step in range(1, math.ceil(env.max_engage_time_s / .1) + 4):
        oa, ob = obs[rows, slots_a], obs[rows, slots_b]
        aa, ha = ma.act(na.normalize(oa) if na else oa, ha,
                        torch.ones_like(starts) if reset_a else starts, sample=False)
        ab, hb = mb.act(nb.normalize(ob) if nb else ob, hb, starts, sample=False)
        controls = torch.empty(n, 2, 4, device="cuda")
        controls[rows, slots_a] = action_to_env(aa, ma.num_bins)
        controls[rows, slots_b] = action_to_env(ab, mb.num_bins)
        obs, reward, done, info = env.step(controls)
        hp = info["terminal_hp"]
        alt = info["terminal_alt_m"]
        hp_a, hp_b = hp[rows, slots_a], hp[rows, slots_b]
        alt_a, alt_b = alt[rows, slots_a], alt[rows, slots_b]
        active = ~finished
        if not bool((info["terminal_state_finite"] | ~active).all()):
            raise FloatingPointError("non-finite raw evaluation trajectory before autoreset")
        # Check raw numeric values, not booleans formed by NaN comparisons.
        require_finite((hp[active], alt[active], reward[active]), "active evaluation transition")
        first_hit = torch.where(active & (first_hit < 0) & (hp_b < 1 - 1e-8),
                                step * .1, first_hit)
        first_hit_b = torch.where(active & (first_hit_b < 0) & (hp_a < 1 - 1e-8),
                                  step * .1, first_hit_b)
        newly = done & active
        if bool(newly.any()):
            da = (hp_a <= 0) | (alt_a < env.min_altitude_m)
            db = (hp_b <= 0) | (alt_b < env.min_altitude_m)
            both_alive = ~da & ~db
            win = (db & ~da) | (both_alive & (hp_a > hp_b + 1e-9))
            lose = (da & ~db) | (both_alive & (hp_a < hp_b - 1e-9))
            score = win.float() + .5 * (~win & ~lose).float()
            indices = newly.nonzero().flatten()
            data = torch.stack((score, hp_a-hp_b, (alt_a < env.min_altitude_m).float(),
                                first_hit, (alt_b < env.min_altitude_m).float(), first_hit_b), 1)[indices].cpu().numpy()
            if not np.isfinite(data).all():
                raise FloatingPointError("non-finite evaluation trajectory")
            for i, values in zip(indices.cpu().tolist(), data):
                records[i] = dict(score=float(values[0]), damage_diff=float(values[1]),
                                  crash=bool(values[2]), first_hit_sec=float(values[3]) if values[3] >= 0 else None,
                                  crash_b=bool(values[4]), first_hit_b_sec=float(values[5]) if values[5] >= 0 else None,
                                  duration_sec=step*.1, ic_pair=i % half, role_swapped=i >= half)
        finished |= newly
        starts = done.float()
        if bool(finished.all()):
            break
    if not bool(finished.all()):
        raise RuntimeError("evaluation lanes did not finish within the episode limit")
    return {"summary": summarize(records), "records": records,
            "wall_sec": time.perf_counter()-started}


def run_suite(spec_path, output, stop_file=None):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    output = Path(output)
    signature = json.dumps(spec, sort_keys=True)
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"spec": spec, "matches": {}}
    if json.dumps(data["spec"], sort_keys=True) != signature:
        raise ValueError("refusing to mix different evaluation manifests")
    torch.set_num_threads(1)
    models = {name: load_policy(path) for name, path in spec["models"].items()}
    env = GpuDogfightVecEnv(spec.get("games", 256), seed=spec["seed"])
    if env.nenv % 2:
        raise ValueError("paired evaluation requires an even number of games")
    for pair in spec["pairs"]:
        an, bn = pair[:2]
        reset = bool(pair[2]) if len(pair) > 2 else False
        for scenario in ("three_nine", "headon"):
            if stop_file and Path(stop_file).exists():
                raise InterruptedError("evaluation STOP requested at complete-match boundary")
            key = f"{an}|{bn}|{scenario}|reset{int(reset)}"
            if key in data["matches"]:
                continue
            print(f"[eval] {key}", flush=True)
            result = evaluate_pair(env, models[an], models[bn], scenario, spec["seed"], reset)
            result.update(a=an, b=bn, scenario=scenario, reset_a=reset)
            data["matches"][key] = result
            atomic_json(output, data)
            print(f"[eval] {key}: {result['summary']}", flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", default=None)
    args = parser.parse_args()
    try:
        run_suite(args.spec, args.output, args.stop_file)
    except InterruptedError as exc:
        print(str(exc), flush=True)
    except (FloatingPointError, AssertionError, ValueError) as exc:
        record_integrity_failure(Path(args.output).parent, exc, "evaluation")
        raise
