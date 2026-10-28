# -*- coding: utf-8 -*-
"""MLP-only GPU PPO (GpuDogfightVecEnv 전용) + opponent pool 자기대전.

GRU factory/step interfaces remain only for historical checkpoint evaluation.
Training uses flat transition minibatches, without recurrent padding or TBPTT.

GPU 벡터 env 위에서 rollout·GAE·업데이트를 GPU 텐서로 계산한다.
Custom kernels use the current Torch stream, without a context barrier at every
launch. PPO remains strictly synchronous: the next rollout starts after update.
At rollout boundaries a finite check rejects invalid physics/training data.

원본과 동일한 부분:
  - **관측** = claude164r(my_observation, 184dim), 수치 회귀 검사 대상.
  - **보상** = my_reward(MY_REWARD_CONFIG), 수치 회귀 검사 대상.
  - **행동공간** = **discrete**: 4채널(roll/pitch/rudder/throttle) × num_bins(기본 21) 균등격자
    linspace(-1,1,21), 채널별 독립 Categorical(model.MLPDiscreteActorCritic 과 동일). 정책 raw
    action(∈[-1,1]^4)→서버 command 변환은 throttle 만 [-1,1]→[0,1](0.5z+0.5), 나머지는 그대로
    (claude_code.action_provider.policy_action_to_command 과 동일). CUDA FCS 는 throttle∈[0,1]
    (thr_pos=2·c_thr, 0.8→afterburner), roll/pitch/rudder∈[-1,1].
  - **초기 상태 분포** = STANDARD_ENV_CONFIG(시나리오 A/B, 고도 2000~30000ft, 속도 200~300m/s).
  - **action_repeat** = step_ratio 6(substeps=6).

opponent pool 자기대전(원본 gated self-play 재현):
  - env 기체0 = **main actor**(학습 정책), 기체1 = **opponent**(pool 에서 env 별 샘플된 frozen
    snapshot). **main 기체 전이만** 학습 버퍼에 담겨 opponent 데이터는 업데이트에서 자동 배제.
  - **EMA 승률 게이팅**: 각 opponent 엔트리는 '우리(main) 승률 EMA'(alpha=selfplay_ema_alpha)를
    가진다. **evictable(net) 후보들의 최소 EMA ≥ gate_threshold** 면 현재 main 을 새 evictable
    snapshot(EMA 0.5)으로 추가(초과 시 oldest evictable FIFO). 고정주기 추가는 없다.
  - **softmax PFSP**: p_i = f/m + (1-f)·softmax(-ema_i/τ). 기본값은 f=0.5, τ=0.3으로,
    어려운(낮은 EMA 승률) 상대를 우대하면서 모든 상대에 충분한 균등 표본을 남긴다.
  - **milestone**(milestone_period): 그때 main 을 permanent(never-evict) 추가 + capacity +1, 이어
    **exploiter** 를 그때 main 유일 상대(frozen)로 scratch 학습(승률 target/max_iters) 후 permanent 추가.

autoreset 계약: HP/고도 종료와 200초 경기 종료는 모두 미래 가치가 0인 task terminal이다.
timeout은 결과 보상 선택을 위해 env에서 truncated로 구분하지만 bootstrap하지 않는다.
진행 중인 경기의 rollout/sequence 경계 bootstrap 및 GAE 연결은 그대로 유지한다.

2026-08-31 fixes: CUDA bundle-specific submission contract, stable per-episode
opponent IDs, deferred retirement and main-runtime preservation across exploiters.
Old search results remain historical/provisional; legacy training resume is rejected
instead of silently mixing protocols. See INTEGRITY_HOLD and the validation report.
"""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from claude_code.model import GRUDiscreteActorCritic, MLPDiscreteActorCritic
from cuda_fdm.finite_checks import require_finite, require_finite_training_stats
from cuda_fdm.future_aux import (AUX_DIM, AUX_FEATURE_DIM, AUX_POS_SCALE_M, AUX_PROTOCOL,
                                 AUX_CONTRACT, build_future_labels, auxiliary_error)
from cuda_fdm.reward_modes import (REWARD_CONTRACT, STANDARD_REWARD,
                                   ALTITUDE_HUNT_REWARD, ATTACK_REWARD, DEFENSE_REWARD,
                                   scheduled_exploiter_mode, reward_mode_name, validate_reward_mode)
from cuda_fdm.league import (ACTIVE_LAYOUT, LEAGUE_PROTOCOL, LeagueArchive,
                             count_aware_alpha, role_mixture,
                             paired_score_interval)

ACTION_BINS = 21   # 원본 train.py --action-bins 기본값과 동일(채널당 21 균등격자).
LEGACY_TRAINING_PROTOCOL = "cuda_mlp_flat_finite_horizon_diverse_h3_reset_v7"
TRAINING_PROTOCOL = "cuda_mlp_flat_finite_horizon_active_league_v16_role_profile_successor"
ACTIVE_AUX_PROTOCOL = TRAINING_PROTOCOL + "+future_position_aux_0p1"
EXPLOITER_ROLES = ("ME-EIE", "LE", "ME-ERE")
# 2026-09-02 (user decision): LE trains against the league mixture (which
# already contains easy/old main clones) but is admitted only against frozen
# current main -- across all 10 LE sessions run so far, every one crossed the
# training win-rate target (0.75-0.80 EMA) yet zero were ever admitted (real
# evaluation against current main, and the LE-only `league_contribution`
# alternate route, both failed every time). Rather than patch the mismatch,
# drop LE from the *scheduled* rotation. `EXPLOITER_ROLES` itself is left
# unchanged -- it is embedded verbatim in `_league_contract()` and checked
# for strict equality on `PPOGPUTrainer.load()`, so changing it would break
# resuming this checkpoint. All LE machinery (the role-profile bandit cell,
# `_train_exploiter_inner`'s LE branches, `league_contribution` admission)
# stays in place and stays valid for any existing LE-labeled archive
# entries; `_exploiter_role()` below just never selects it going forward.
EXPLOITER_ACTIVE_ROLES = ("ME-EIE", "ME-ERE")
PROFILE_NAMES = ("standard", "altitude_hunt", "attack", "defense")
# P0 fix (2026-09-02, v17): the wr_ema<0.20 curriculum bailout used to
# re-decide the opponent category every single iteration (`i % 2 == 0`).
# Once the reset-on-category-change fix landed (every switch now pays a full
# env reset), a stuck exploiter that hovers right at the 0.20 boundary
# toggled category on almost every iteration -- a full reset every iteration,
# and a win-rate graph that checkerboards between an easy and a hard
# opponent rather than showing a trend. Widen the alternation from 1
# iteration to a block of this many, so the category decision below only
# changes at most once every `EXPLOITER_CURRICULUM_BLOCK_ITERS` iterations,
# long enough to both amortize the reset cost and read as a trend instead of
# noise.
EXPLOITER_CURRICULUM_BLOCK_ITERS = 6
# 2026-09-03 (user decision): replace the hard wr_ema<0.20 on/off gate with a
# soft probability band. The hard threshold made a struggling exploiter swap
# its *entire* opponent between "frozen main" and "easy curriculum" the
# instant wr_ema crossed 0.20 in either direction, which for an exploiter
# hovering right at the boundary meant blocks of curriculum and blocks of
# frozen-main alternated in lockstep with small wr_ema noise (confirmed in
# production logs: 25-32% of a session's iterations spent in curriculum,
# oscillating rather than trending). A linear probability ramp over a band
# instead of a single cutoff means nearby wr_ema values get nearby curriculum
# odds, so the mix drifts smoothly as the exploiter's real performance
# changes rather than flipping a coin the instant it crosses one point.
EXPLOITER_CURRICULUM_BAND_LOW = 0.10   # wr_ema <= this: always curriculum (p=1)
EXPLOITER_CURRICULUM_BAND_HIGH = 0.30  # wr_ema >= this: never curriculum (p=0)
PROFILE_SCHEDULER_PROTOCOL = "role_profile_bandit_v1"


def _curriculum_probability(wr_ema: float) -> float:
    """Linear ramp: 1.0 at/below BAND_LOW, 0.0 at/above BAND_HIGH."""
    lo, hi = EXPLOITER_CURRICULUM_BAND_LOW, EXPLOITER_CURRICULUM_BAND_HIGH
    if wr_ema <= lo:
        return 1.0
    if wr_ema >= hi:
        return 0.0
    return (hi - wr_ema) / (hi - lo)


def _empty_role_profile_bandit():
    return {
        role: {name: {"count": 0, "utility": 0.0} for name in PROFILE_NAMES}
        for role in EXPLOITER_ROLES
    }


def _record_role_profile_result(state, role, profile, target_score, novelty):
    if role not in EXPLOITER_ROLES or profile not in PROFILE_NAMES:
        raise ValueError(f"unknown exploiter role/profile: {role}/{profile}")
    target_score = float(target_score)
    novelty = float(novelty)
    if not math.isfinite(target_score) or not math.isfinite(novelty):
        raise FloatingPointError("non-finite role/profile scheduler evidence")
    item = state[role][profile]
    item["count"] = int(item["count"]) + 1
    utility = target_score + 0.25 * novelty
    item["utility"] = float(item["utility"] + (utility - item["utility"]) / item["count"])


def _validate_role_profile_bandit(state):
    if set(state) != set(EXPLOITER_ROLES):
        raise ValueError("role/profile scheduler roles differ")
    result = _empty_role_profile_bandit()
    for role in EXPLOITER_ROLES:
        if set(state[role]) != set(PROFILE_NAMES):
            raise ValueError(f"role/profile scheduler profiles differ for {role}")
        for profile in PROFILE_NAMES:
            item = state[role][profile]
            count = int(item["count"])
            utility = float(item["utility"])
            if count < 0 or not math.isfinite(utility):
                raise ValueError("invalid role/profile scheduler state")
            result[role][profile] = {"count": count, "utility": utility}
    return result


def role_profile_bandit_from_history(history):
    """Rebuild the exact per-role scheduler evidence from admission history."""
    state = _empty_role_profile_bandit()
    for row in history:
        admission = row.get("admission") or {}
        role = row.get("role")
        profile = row.get("reward_mode", row.get("profile"))
        if (role not in EXPLOITER_ROLES or profile not in PROFILE_NAMES
                or "target_score" not in admission or "payoff_novelty" not in admission):
            raise ValueError("exploiter history cannot reconstruct role/profile scheduler")
        _record_role_profile_result(
            state, role, profile, admission["target_score"], admission["payoff_novelty"])
    return state


def make_action_grid(num_bins=ACTION_BINS, device="cpu"):
    """[-1,1] 균등 num_bins 격자(양끝 포함). 홀수면 가운데=정확히 0(중립)."""
    return torch.linspace(-1.0, 1.0, int(num_bins), device=device)


# ── 관측 running 정규화 (GPU, 배치 병렬분산) ──────────────────────────────────
class RunningNorm:
    def __init__(self, dim, device, clip=10.0, eps=1e-8):
        self.dim = dim
        self.device = device
        self.mean = torch.zeros(dim, device=device, dtype=torch.float32)
        self.var = torch.ones(dim, device=device, dtype=torch.float32)
        self.count = torch.zeros((), device=device, dtype=torch.float32) + 1e-4
        self.clip = float(clip)
        self.eps = float(eps)

    @torch.no_grad()
    def update(self, x):
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        b_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + (delta * delta) * (self.count * b_count / tot)
        self.var = M2 / tot
        self.count = tot

    @torch.no_grad()
    def normalize(self, x):
        n = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return n.clamp_(-self.clip, self.clip)

    def state_dict(self):
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count.clone()}

    def load_state_dict(self, sd):
        self.mean.copy_(sd["mean"]); self.var.copy_(sd["var"]); self.count.copy_(sd["count"])

    def clone(self):
        c = RunningNorm(self.dim, self.device, self.clip, self.eps)
        c.load_state_dict(self.state_dict())
        return c


# ── actor-critic (independent actor/critic GRU) ───────────────────────────────
class ActorCritic(GRUDiscreteActorCritic):
    """CUDA PPO용 recurrent 정책.

    관측 184D 뒤에 actor/critic이 서로 독립된 ``512→512→GRU(256)→512`` 경로를
    갖는다. 구현과 state_dict 이름을 CPU/제출용 ``GRUDiscreteActorCritic``과 공유해
    CUDA 체크포인트를 별도 근사 변환 없이 그대로 bundle로 옮길 수 있다.
    """

    def __init__(self, obs_dim, act_dim=4, num_bins=ACTION_BINS,
                 hidden=(512, 512, 512), activation="tanh", gru_size=256, encoder_depth=2):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, num_bins=num_bins,
                         hidden=hidden, activation=activation, gru_size=gru_size,
                         encoder_depth=encoder_depth)
        self.state_size = int(gru_size)

    @torch.no_grad()
    def act(self, obs, state=None, episode_start=None, sample=True):
        """행동 index와 갱신된 recurrent state를 함께 반환한다."""
        action, _, _, state = self.actor_step(
            obs, state=state, episode_start=episode_start,
            deterministic=not sample)
        return action.long(), state


class MLPActorCritic(MLPDiscreteActorCritic):
    """CUDA PPO용 feed-forward 정책.

    과거 checkpoint의 rollout/pool/evaluation 인터페이스를 읽기 위해 크기 1의
    무의미한 state를 제공한다. MLP updater는 state나 sequence를 사용하지 않는다.
    네트워크 계산에는 이 state가 전혀
    사용되지 않으므로 일반 ``MLPDiscreteActorCritic``과 정확히 같은 정책이다.
    """

    is_recurrent = False
    state_size = 1

    def __init__(self, obs_dim, act_dim=4, num_bins=ACTION_BINS,
                 hidden=(512, 512, 512), activation="tanh", gru_size=None, encoder_depth=2):
        del gru_size
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, num_bins=num_bins,
                         hidden=hidden, activation=activation)

    def initial_state(self, batch_size: int, device=None):
        device = device if device is not None else next(self.parameters()).device
        z = torch.zeros(1, int(batch_size), 1, device=device)
        return z, z.clone()

    def actor_step(self, obs, state=None, episode_start=None, action=None,
                   deterministic: bool = False):
        del episode_start
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        logits = self.actor_logits(obs).view(-1, self.act_dim, self.num_bins)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if deterministic else dist.sample()
        else:
            action = action.long()
        return (action.float(), dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1), state)

    def value_step(self, obs, state=None, episode_start=None):
        del episode_start
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        return self.get_value(obs), state

    @torch.no_grad()
    def act(self, obs, state=None, episode_start=None, sample=True):
        action, _, _, state = self.actor_step(
            obs, state=state, episode_start=episode_start,
            deterministic=not sample)
        return action.long(), state

class AuxiliaryMLPActorCritic(MLPActorCritic):
    """The same inference MLPs with two training-only six-dimensional heads.

Existing actor_logits/critic state-dict names and rollout forwards are unchanged.
PPO and auxiliary heads share each network's trunk, not actor/critic parameters.
    """
    uses_future_aux = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Constructors run on CPU before .to(device). Do not change the base
        # model/pool sampling RNG just because optional heads were initialized.
        with torch.random.fork_rng(devices=[]):
            self.actor_aux_head = nn.Linear(self.actor_logits[-1].in_features, AUX_DIM)
            self.critic_aux_head = nn.Linear(self.critic[-1].in_features, AUX_DIM)
            for head in (self.actor_aux_head, self.critic_aux_head):
                nn.init.orthogonal_(head.weight, gain=0.01)
                nn.init.zeros_(head.bias)
        # Plain tuples: reuse layers without registering duplicate parameters or
        # constructing Sequential slices in every minibatch.
        self._actor_feature_layers = tuple(self.actor_logits.children())[:-1]
        self._critic_feature_layers = tuple(self.critic.children())[:-1]

    @staticmethod
    def _features(layers, obs):
        for layer in layers:
            obs = layer(obs)
        return obs

    def actor_parameters(self):
        return super().actor_parameters() + list(self.actor_aux_head.parameters())

    def critic_parameters(self):
        return super().critic_parameters() + list(self.critic_aux_head.parameters())

    def evaluate_actions_with_aux(self, obs, action):
        features = self._features(self._actor_feature_layers, obs)
        logits = self.actor_logits[-1](features).view(-1, self.act_dim, self.num_bins)
        dist = torch.distributions.Categorical(logits=logits)
        return (dist.log_prob(action.long()).sum(-1), dist.entropy().sum(-1),
                self.actor_aux_head(features))

    def value_with_aux(self, obs):
        features = self._features(self._critic_feature_layers, obs)
        return self.critic[-1](features).squeeze(-1), self.critic_aux_head(features)


def build_actor_critic(**model_kwargs):
    """구조 이름으로 GRU/MLP를 생성하는 단일 factory."""
    kwargs = dict(model_kwargs)
    architecture = str(kwargs.pop("architecture", "gru")).lower()
    aux_pred = bool(kwargs.pop("aux_pred", False))
    if architecture == "gru":
        if aux_pred:
            raise ValueError("future-position auxiliary training is MLP-only")
        return ActorCritic(**kwargs)
    if architecture == "mlp":
        return (AuxiliaryMLPActorCritic if aux_pred else MLPActorCritic)(**kwargs)
    raise ValueError(f"unknown architecture: {architecture!r}")


def action_to_env(idx, num_bins, grid=None):
    """행동 index(N,4) → env control(N,4). 격자 linspace(-1,1,num_bins) 로 연속화한 뒤
    throttle 만 [-1,1]→[0,1](0.5z+0.5), roll/pitch/rudder 는 그대로([-1,1])."""
    if grid is None:
        grid = make_action_grid(num_bins, device=idx.device)
    cont = grid[idx.long()]                                  # (N,4) in [-1,1]
    ctrl = cont.clone()
    ctrl[:, 3] = 0.5 * cont[:, 3] + 0.5                      # throttle → [0,1]
    return ctrl


# ── opponent pool (EMA 게이팅 + softmax 가중 샘플링) ──────────────────────────
class OpponentPool:
    """Original PFSP over active snapshots; stable IDs for entire episodes.

    FIFO eviction retires an entry from *sampling*, not from an ongoing game.
    Retired networks are released only after their last assigned episode ends.
    Entries may therefore temporarily exceed the sampling capacity.
    """

    def __init__(self, model_kwargs, device, evict_cap=4, sample=True):
        self.model_kwargs = dict(model_kwargs)
        self.device = device
        self.evict_cap = int(evict_cap)
        if self.evict_cap < 1:
            raise ValueError("pool evictable capacity must be positive")
        self.sample = bool(sample)
        self.entries = []
        self.next_id = 0
        self._reindex()

    def active_entries(self):
        return [e for e in self.entries if not e["retired"]]

    def size(self):
        return len(self.active_entries())

    def resident_size(self):
        return len(self.entries)

    def _reindex(self):
        self.ids = torch.tensor([e["id"] for e in self.entries],
                                dtype=torch.long, device=self.device)
        rows = [-1] * self.next_id
        for row, e in enumerate(self.entries):
            rows[e["id"]] = row
        self._row_by_id = torch.tensor(rows, dtype=torch.long, device=self.device)

    def rows_for_ids(self, ids):
        # No clamp: invalid identities must not silently become another opponent.
        return self._row_by_id.index_select(0, ids)

    def refresh_residents(self, assignments, done):
        """Iteration boundary only: validate IDs and release unreferenced retirees.

        A single small host copy is needed only while entries are retiring.
        Fixed lane lists let retired networks infer only their remaining games,
        without a GPU nonzero()/host synchronization on every rollout step.
        """
        if not any(e["retired"] for e in self.entries):
            return
        a = assignments.detach().cpu().numpy()
        live = ~done.detach().bool().cpu().numpy()
        live_ids = set(a[live].tolist())
        resident_ids = {e["id"] for e in self.entries}
        if not live_ids.issubset(resident_ids):
            raise ValueError("live episode refers to a missing opponent ID")
        kept = []
        for e in self.entries:
            if e["retired"]:
                lanes = np.flatnonzero(live & (a == e["id"]))
                if not len(lanes):
                    continue
                e["retired_lanes"] = torch.as_tensor(lanes, device=self.device)
            kept.append(e)
        changed = len(kept) != len(self.entries)
        self.entries = kept
        if changed:
            self._reindex()

    def num_permanent(self):
        return sum(1 for e in self.entries if e["permanent"])

    def capacity(self):
        return self.evict_cap + self.num_permanent()

    def slot_emas(self):
        """슬롯별 EMA 리포팅용. entries 는 insert 순서라 non-permanent 만 뽑으면 FIFO 순서
        (index 0=가장 먼저 들어온 evictable). 반환 (evict_emas, perm_emas):
          - evict_emas[i] = i 번째 evict 슬롯의 현재 opponent EMA. 새 opponent 가 들어와
            oldest 가 빠지면 슬롯 i 는 '다음으로 오래된 opponent' 로 바뀔 뿐 슬롯 수(≤evict_cap)
            는 안 늘어난다 → wandb 그래프가 opponent 마다 늘어나지 않고 슬롯 단위로 고정.
          - perm_emas[i] = i 번째 permanent(never-evict) opponent EMA(추가 순, 안정적 identity).
        EMA 계산 자체는 opponent 별(update_emas)로 그대로 유지, 여기선 리포팅 매핑만 한다."""
        evict_emas = [e["ema"] for e in self.active_entries() if not e["permanent"]]
        perm_emas = [e["ema"] for e in self.entries if e["permanent"]]
        return evict_emas, perm_emas

    def _mk_net(self, model):
        require_finite(model.state_dict(), "new opponent parameters")
        from cuda_fdm.external_opponents import ExternalActor
        if isinstance(model, ExternalActor):
            return copy.deepcopy(model).to(self.device).eval()
        net = build_actor_critic(**self.model_kwargs).to(self.device)
        net.load_state_dict(copy.deepcopy(model.state_dict()))
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    @torch.no_grad()
    def add(self, model, norm, permanent, ema=0.5):
        entry = {"net": self._mk_net(model), "norm": norm.clone() if norm is not None else None,
                 "id": self.next_id, "retired": False,
                 "permanent": bool(permanent), "ema": float(ema), "actor_state": None}
        self.next_id += 1
        self.entries.append(entry)
        # Retire the same FIFO sampling slot; do not mutate any live assignment.
        ev = [e for e in self.active_entries() if not e["permanent"]]
        while len(ev) > self.evict_cap:
            ev.pop(0)["retired"] = True
        self._reindex()
        return entry["id"]

    def update_emas(self, win_by_opp, loss_by_opp, ep_by_opp, alpha, opponent_ids=None):
        """이번 iteration 의 per-opponent 결과로 각 엔트리 EMA 갱신.
        완료 에피소드가 하나라도 있으면(ep>0) 무승부를 0.5 로 반영해 갱신한다:
        frac = (승 + 0.5·무) / 완료 = (w + 0.5·(ep - w - l)) / ep.  ep==0(그 iter 에 이
        opponent 가 샘플링 안 됐거나 결판/무승부 모두 없음)일 때만 이전 EMA 를 유지한다.
        (예전엔 '결판(win|loss)>0' 일 때만 갱신 → 무승부만 난 iter 는 EMA 가 안 움직여
        그래프가 가로 일직선으로 남았음. 이제 무승부도 0.5 로 EMA 를 끌어당긴다.)"""
        w = win_by_opp.detach().cpu().numpy()
        l = loss_by_opp.detach().cpu().numpy()
        ep = ep_by_opp.detach().cpu().numpy()
        ids = ([e["id"] for e in self.entries] if opponent_ids is None
               else opponent_ids.detach().cpu().tolist())
        if len(ids) != len(w) or len(w) != len(l) or len(w) != len(ep):
            raise ValueError("opponent result shape/identity mismatch")
        by_id = {e["id"]: e for e in self.entries}
        for i, identity in enumerate(ids):
            if identity not in by_id:
                raise ValueError(f"results refer to missing opponent ID {identity}")
            e = by_id[identity]
            if ep[i] > 0:
                frac = (w[i] + 0.5 * (ep[i] - w[i] - l[i])) / ep[i]
                e["ema"] = float((1.0 - alpha) * e["ema"] + alpha * frac)

    def gate_and_add(self, model, norm, threshold):
        """evictable(net) 후보 최소 EMA ≥ threshold 면 현재 main 을 새 evictable 로 추가."""
        net_emas = [e["ema"] for e in self.active_entries() if not e["permanent"]]
        if net_emas and min(net_emas) >= threshold:
            self.add(model, norm, permanent=False, ema=0.5)
            return True
        return False

    @torch.no_grad()
    def weights(self, temp, floor):
        """원래 PFSP: f/m + (1-f)·softmax(-EMA/temperature)."""
        active_rows = [i for i, e in enumerate(self.entries) if not e["retired"]]
        emas = np.array([self.entries[i]["ema"] for i in active_rows], dtype=np.float64)
        m = emas.size
        if not m:
            raise ValueError("opponent pool has no active sampling entries")
        logits = -emas / max(float(temp), 1.0e-6)
        logits -= logits.max()
        priority = np.exp(logits)
        priority /= priority.sum()
        floor = float(np.clip(floor, 0.0, 1.0))
        p = floor / m + (1.0 - floor) * priority
        p /= p.sum()
        resident_p = np.zeros(len(self.entries), dtype=np.float64)
        resident_p[active_rows] = p
        return torch.as_tensor(resident_p, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, opp_obs, assign, episode_start):
        """Infer assigned MLP rows; assign contains stable episode IDs.

        Only the expensive active-MLP forward/normalization is partitioned.
        Full-shaped sampling, including unused rows, preserves the random-number
        stream of the old path. Retired lane lists and legacy GRU state updates
        keep their existing behavior. A one-opponent pool needs no partition.
        """
        rows = self.rows_for_ids(assign)
        groups = None
        if len(self.entries) > 1 and not any(e["net"].is_recurrent for e in self.entries):
            # One small host transfer per step, not one nonzero()/sync per
            # opponent. Recompute after episode-end reassignment; no stale cache.
            counts = torch.bincount(rows, minlength=len(self.entries)).cpu().tolist()
            groups = torch.argsort(rows, stable=True).split(counts)
        outs = []
        for row, e in enumerate(self.entries):
            if e.get("actor_state") is None or e["actor_state"][0].shape[1] != opp_obs.shape[0]:
                e["actor_state"] = e["net"].initial_state(opp_obs.shape[0], self.device)
            if groups is not None and not e["retired"]:
                net, lanes = e["net"], groups[row]
                logits = opp_obs.new_zeros((opp_obs.shape[0], net.act_dim, net.num_bins))
                if lanes.numel():
                    oo = opp_obs.index_select(0, lanes)
                    on = e["norm"].normalize(oo) if e["norm"] is not None else oo
                    part = net.actor_logits(on).view(-1, net.act_dim, net.num_bins)
                    logits.index_copy_(0, lanes, part)
                # Do not sample a smaller tensor or skip empty entries: either
                # would change the learner's subsequent RNG stream as well.
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample() if self.sample else logits.argmax(-1)
                outs.append(action)
                continue
            lanes = e.get("retired_lanes") if e["retired"] else None
            oo = opp_obs if lanes is None else opp_obs.index_select(0, lanes)
            on = e["norm"].normalize(oo) if e["norm"] is not None else oo
            hidden = e["actor_state"]
            starts = episode_start
            if lanes is not None:
                hidden = tuple(h.index_select(1, lanes) for h in hidden)
                starts = None if starts is None else starts.index_select(0, lanes)
            action, state = e["net"].act(
                on, state=hidden, episode_start=starts,
                sample=self.sample)
            if lanes is None:
                e["actor_state"] = tuple(x.detach() for x in state)
            else:
                for full, part in zip(e["actor_state"], state):
                    full.index_copy_(1, lanes, part.detach())
                full_action = torch.zeros(opp_obs.shape[0], action.shape[-1],
                                          dtype=action.dtype, device=action.device)
                full_action.index_copy_(0, lanes, action)
                action = full_action
            outs.append(action)                                  # (nenv,4) long
        stacked = torch.stack(outs, 0)                            # (P,nenv,4)
        idx = rows.view(1, -1, 1).expand(1, opp_obs.shape[0], 4)
        return stacked.gather(0, idx).squeeze(0)

    def state_dicts(self):
        out = []
        for e in self.entries:
            out.append({"model": {k: v.detach().cpu() for k, v in e["net"].state_dict().items()},
                        "norm": (e["norm"].state_dict() if e["norm"] is not None else None),
                        "id": e["id"], "retired": e["retired"],
                        "permanent": e["permanent"], "ema": e["ema"]})
        return out

    def load_state_dicts(self, dicts, next_id=None):
        ids = [d.get("id") for d in dicts]
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("checkpoint must contain unique stable opponent IDs")
        self.next_id = max(ids) + 1 if next_id is None else int(next_id)
        if self.next_id <= max(ids):
            raise ValueError("opponent next_id would reuse an existing ID")
        self.entries = []
        for d in dicts:
            net = build_actor_critic(**self.model_kwargs).to(self.device)
            net.load_state_dict({k: torch.as_tensor(v) for k, v in d["model"].items()})
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            norm = None
            if d["norm"] is not None:
                norm = RunningNorm(self.model_kwargs["obs_dim"], self.device)
                norm.load_state_dict({k: torch.as_tensor(v, device=self.device)
                                      for k, v in d["norm"].items()})
            self.entries.append({"net": net, "norm": norm,
                                 "id": d["id"], "retired": bool(d["retired"]),
                                 "permanent": bool(d["permanent"]),
                                 "ema": float(d.get("ema", 0.5)), "actor_state": None})
        self._reindex()


class ActiveLeaguePool(OpponentPool):
    """GPU-resident subset of a disk-backed league.

    Opponent IDs remain stable for an episode.  Re-selection only retires an
    entry from new matchmaking; a resident is released after its final live
    episode.  Stochastic opponent actions use per-policy generators, allowing
    inference only on assigned lanes without consuming the learner RNG stream.
    """

    def __init__(self, model_kwargs, device, active_cap=24, sample=True,
                 ema_half_life_games=512.0, non_latest_cap=0.12, seed=0):
        self.active_cap = int(active_cap)
        if self.active_cap < sum(ACTIVE_LAYOUT.values()):
            raise ValueError("active league cap must be at least 24")
        self.ema_half_life_games = float(ema_half_life_games)
        self.non_latest_cap = float(non_latest_cap)
        self.seed = int(seed)
        super().__init__(model_kwargs, device, evict_cap=self.active_cap, sample=sample)

    def capacity(self):
        return self.active_cap

    def num_permanent(self):
        # Kept for the old logging field; permanence now belongs to the archive.
        return 0

    @staticmethod
    def _normalise_probability(values, *, fallback=None, label="sampling weights"):
        """Return a finite probability vector, optionally using an explicit fallback.

        League channels are assembled on the CPU and copied to CUDA only after
        validation.  This keeps malformed/empty early-league channels away from
        ``torch.multinomial``: a NaN probability otherwise raises an asynchronous
        device-side assert and poisons the entire CUDA context.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not values.size:
            raise ValueError(f"{label} must be a non-empty vector")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{label} contains non-finite or negative values")
        total = float(values.sum())
        if total <= 0.0:
            if fallback is None:
                raise ValueError(f"{label} has no positive mass")
            values = np.asarray(fallback, dtype=np.float64)
            if (values.shape != (values.size,) or not values.size
                    or not np.isfinite(values).all() or np.any(values < 0.0)):
                raise ValueError(f"{label} fallback is invalid")
            total = float(values.sum())
            if total <= 0.0:
                raise ValueError(f"{label} fallback has no positive mass")
        return values / total

    def role_counts(self):
        return {role: sum(e.get("role") == role for e in self.active_entries())
                for role in ACTIVE_LAYOUT}

    def _new_generator(self, identity, state=None):
        gen = torch.Generator(device=self.device)
        if state is None:
            gen.manual_seed((self.seed * 1_000_003 + int(identity) * 97_409 + 17) % (2**63 - 1))
        else:
            gen.set_state(torch.as_tensor(state, dtype=torch.uint8, device="cpu"))
        return gen

    @torch.no_grad()
    def add(self, model, norm, permanent=False, ema=0.5, *, role="core", archive_id=None,
            created_iteration=0, profile="standard", coverage=False, past_best=None,
            games=0.0, nash_mass=0.0, rng_state=None):
        if role not in ACTIVE_LAYOUT:
            raise ValueError(f"unknown active league role: {role}")
        entry = {"net": self._mk_net(model), "norm": norm.clone() if norm is not None else None,
                 "id": self.next_id, "retired": False, "permanent": False,
                 "ema": float(ema), "past_best": float(ema if past_best is None else past_best),
                 "games": float(games), "actor_state": None, "role": role,
                 "archive_id": None if archive_id is None else int(archive_id),
                 "created_iteration": int(created_iteration), "profile": str(profile),
                 "coverage": bool(coverage), "nash_mass": float(nash_mass)}
        entry["rng"] = self._new_generator(entry["id"], rng_state)
        self.next_id += 1
        self.entries.append(entry)
        self._reindex()
        self._validate_layout()
        return entry["id"]

    def _validate_layout(self):
        active = self.active_entries()
        if len(active) > self.active_cap:
            raise ValueError(f"active league exceeds cap: {len(active)} > {self.active_cap}")
        counts = self.role_counts()
        for role, limit in ACTIVE_LAYOUT.items():
            if counts[role] > limit:
                raise ValueError(f"active role {role} exceeds quota {limit}")

    def retire(self, entry):
        entry["retired"] = True

    def replace_latest(self, model, norm, iteration):
        for entry in self.active_entries():
            if entry.get("role") == "latest":
                self.retire(entry)
        identity = self.add(model, norm, role="latest", created_iteration=iteration)
        self._reindex()
        return identity

    def add_recent_bundle(self, model, norm, archive_id, iteration, profile="standard"):
        recent = sorted((e for e in self.active_entries() if e.get("role") == "recent"),
                        key=lambda e: e["created_iteration"])
        if len(recent) >= ACTIVE_LAYOUT["recent"]:
            # Recent is a temporal curriculum ring, not a mastery gate.  Keep
            # strict chronology; difficulty/payoff policies belong in core.
            self.retire(recent[0])
        self.add(model, norm, role="recent", archive_id=archive_id,
                 created_iteration=iteration, profile=profile)
        self._reindex(); self._validate_layout()

    def _norm_from_state(self, state):
        if state is None:
            return None
        norm = RunningNorm(self.model_kwargs["obs_dim"], self.device)
        norm.load_state_dict({k: torch.as_tensor(v, device=self.device) for k, v in state.items()})
        return norm

    def add_archive_policy(self, archive, archive_id, role, *, coverage=False):
        # Role-scoped: a milestone-reused archive_id (see the milestone/recent
        # dedup fix) can legitimately be resident twice under different roles
        # (e.g. still-chronological "recent" and newly-selected "core") for a
        # bounded window. Guard only against re-adding the same (id, role).
        if any(e.get("archive_id") == int(archive_id) and e.get("role") == role
               and not e["retired"] for e in self.entries):
            return None
        bundle = archive.load_policy(int(archive_id))
        from cuda_fdm.external_opponents import is_external, frozen_external
        if is_external(bundle):
            net, norm, _ = frozen_external(bundle, self.device)
        else:
            net = build_actor_critic(**self.model_kwargs).to(self.device)
            net.load_state_dict(bundle["model"])
            norm = self._norm_from_state(bundle.get("norm"))
        record = archive.records[int(archive_id)]
        return self.add(net, norm, role=role, archive_id=archive_id,
                        created_iteration=record["iteration"], profile=record.get("profile", "standard"),
                        coverage=coverage, ema=record.get("current_score", 0.5),
                        past_best=record.get("past_best", 0.5),
                        nash_mass=record.get("nash_mass", 0.0))

    def sync_archive_roles(self, archive, core_ids, challenger_ids, coverage_ids=()):
        desired = {int(x): "core" for x in core_ids}
        desired.update({int(x): "challenger" for x in challenger_ids})
        coverage_ids = {int(x) for x in coverage_ids}
        for entry in self.active_entries():
            if entry.get("role") in ("core", "challenger"):
                archive_id = entry.get("archive_id")
                if archive_id not in desired or desired[archive_id] != entry["role"]:
                    self.retire(entry)
                else:
                    entry["coverage"] = archive_id in coverage_ids
        for archive_id, role in desired.items():
            self.add_archive_policy(archive, archive_id, role, coverage=archive_id in coverage_ids)
        self._reindex(); self._validate_layout()

    def update_emas(self, win_by_opp, loss_by_opp, ep_by_opp, alpha=None, opponent_ids=None):
        w = win_by_opp.detach().cpu().numpy()
        l = loss_by_opp.detach().cpu().numpy()
        ep = ep_by_opp.detach().cpu().numpy()
        ids = ([e["id"] for e in self.entries] if opponent_ids is None
               else opponent_ids.detach().cpu().tolist())
        if not (len(ids) == len(w) == len(l) == len(ep)):
            raise ValueError("opponent result shape/identity mismatch")
        by_id = {e["id"]: e for e in self.entries}
        for index, identity in enumerate(ids):
            entry = by_id.get(identity)
            if entry is None:
                raise ValueError(f"results refer to missing opponent ID {identity}")
            games = float(ep[index])
            if games <= 0.0:
                continue
            score = float((w[index] + 0.5 * (ep[index] - w[index] - l[index])) / ep[index])
            coefficient = count_aware_alpha(games, self.ema_half_life_games)
            entry["ema"] = (1.0 - coefficient) * entry["ema"] + coefficient * score
            entry["past_best"] = max(entry["past_best"], entry["ema"])
            entry["games"] += games

    def gate_and_add(self, model, norm, threshold):
        # Recent snapshots have an explicit independent-evaluation gate.
        return False

    @torch.no_grad()
    def weights(self, temp=None, floor=None):
        active = self.active_entries()
        values = self._normalise_probability(
            role_mixture(active, self.non_latest_cap), label="main role mixture")
        resident = np.zeros(len(self.entries), dtype=np.float64)
        active_rows = [i for i, e in enumerate(self.entries) if not e["retired"]]
        resident[active_rows] = values
        return torch.as_tensor(resident, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def variance_curriculum_weights(self):
        active = self.active_entries()
        if not active:
            raise ValueError("variance curriculum requires an active opponent")
        eligible = np.asarray([e.get("role") in ("recent", "core", "challenger") for e in active])
        q = np.asarray([float(e.get("ema", 0.5)) for e in active])
        raw = np.where(eligible, q * (1.0 - q), 0.0)
        raw = self._normalise_probability(
            raw, fallback=(eligible.astype(np.float64) if eligible.any()
                           else np.ones(len(active), dtype=np.float64)),
            label="variance curriculum")
        resident = np.zeros(len(self.entries), dtype=np.float64)
        resident[[i for i, e in enumerate(self.entries) if not e["retired"]]] = raw
        return torch.as_tensor(resident, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def league_exploiter_weights(self):
        active = self.active_entries()
        if not active:
            raise ValueError("league exploiter requires an active opponent")
        nash = np.asarray([float(e.get("nash_mass", 0.0)) for e in active])
        q = np.asarray([float(e.get("ema", 0.5)) for e in active])
        core = np.asarray([e.get("role") in ("core", "challenger") for e in active])
        coverage = np.asarray([bool(e.get("coverage", False)) for e in active])
        # During the first LE milestone there may not yet be a strategic core.
        # In that warm-up state, latest/recent policies are the whole league and
        # form the explicit fallback support for all three LE channels.
        eligible = core if core.any() else np.ones(len(active), dtype=bool)
        hard = np.where(eligible, np.square(1.0 - q), 0.0)
        payoff_coverage = np.where(core & coverage, 1.0, 0.0)
        fallback = eligible.astype(np.float64)
        nash = self._normalise_probability(nash, fallback=fallback,
                                           label="league exploiter Nash channel")
        hard = self._normalise_probability(hard, fallback=fallback,
                                           label="league exploiter hard channel")
        payoff_coverage = self._normalise_probability(
            payoff_coverage, fallback=fallback,
            label="league exploiter payoff-response coverage channel")
        # Each channel is normalised independently.  Active Nash support can
        # sum to less than one because some support policies live only in the
        # cold archive; dividing by max(sum, 1) would silently shrink the 50%
        # Nash channel and inflate the other two after the final normalisation.
        raw = self._normalise_probability(
            0.50 * nash + 0.25 * hard + 0.25 * payoff_coverage,
            label="league exploiter mixture")
        resident = np.zeros(len(self.entries), dtype=np.float64)
        resident[[i for i, e in enumerate(self.entries) if not e["retired"]]] = raw
        return torch.as_tensor(resident, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, opp_obs, assign, episode_start, external_obs=None):
        rows = self.rows_for_ids(assign)
        counts = torch.bincount(rows, minlength=len(self.entries)).cpu().tolist()
        groups = torch.argsort(rows, stable=True).split(counts)
        output = torch.zeros((opp_obs.shape[0], 4), dtype=torch.long, device=opp_obs.device)
        for row, entry in enumerate(self.entries):
            lanes = groups[row]
            if not lanes.numel():
                continue
            source = external_obs if getattr(entry['net'], 'obs_dim', 184)==214 else opp_obs
            if source is None:
                raise ValueError('external pool opponent has no 214D observations')
            obs = source.index_select(0, lanes)
            normed = entry["norm"].normalize(obs) if entry["norm"] is not None else obs
            logits = entry["net"].actor_logits(normed).view(-1, 4, entry["net"].num_bins)
            if self.sample:
                probs = torch.softmax(logits.float(), dim=-1).flatten(0, 1)
                actions = torch.multinomial(probs, 1, generator=entry["rng"]).view(-1, 4)
            else:
                actions = logits.argmax(-1)
            output.index_copy_(0, lanes, actions)
        return output

    def state_dicts(self):
        out = []
        for entry in self.entries:
            out.append({"model": {k: v.detach().cpu() for k, v in entry["net"].state_dict().items()},
                        "norm": entry["norm"].state_dict() if entry["norm"] is not None else None,
                        "id": entry["id"], "retired": entry["retired"], "permanent": False,
                        "ema": entry["ema"], "past_best": entry["past_best"],
                        "games": entry["games"], "role": entry["role"],
                        "archive_id": entry["archive_id"],
                        "created_iteration": entry["created_iteration"],
                        "profile": entry["profile"], "coverage": entry["coverage"],
                        "nash_mass": entry.get("nash_mass", 0.0),
                        "rng_state": entry["rng"].get_state().cpu()})
        return out

    def load_state_dicts(self, dicts, next_id=None):
        ids = [d.get("id") for d in dicts]
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("checkpoint must contain unique stable opponent IDs")
        self.next_id = max(ids) + 1 if next_id is None else int(next_id)
        if self.next_id <= max(ids):
            raise ValueError("opponent next_id would reuse an existing ID")
        self.entries = []
        for value in dicts:
            from cuda_fdm.external_opponents import is_external, frozen_external
            if is_external(value):
                net, norm, _ = frozen_external(value, self.device)
            else:
                net = build_actor_critic(**self.model_kwargs).to(self.device)
                net.load_state_dict(value["model"]); net.eval()
                norm = self._norm_from_state(value.get("norm"))
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            entry = {"net": net, "norm": norm, "id": value["id"],
                     "retired": bool(value["retired"]), "permanent": False,
                     "ema": float(value.get("ema", 0.5)),
                     "past_best": float(value.get("past_best", value.get("ema", 0.5))),
                     "games": float(value.get("games", 0.0)), "actor_state": None,
                     "role": value["role"], "archive_id": value.get("archive_id"),
                     "created_iteration": int(value.get("created_iteration", 0)),
                     "profile": value.get("profile", "standard"),
                     "coverage": bool(value.get("coverage", False)),
                     "nash_mass": float(value.get("nash_mass", 0.0))}
            entry["rng"] = self._new_generator(entry["id"], value.get("rng_state"))
            self.entries.append(entry)
        self._reindex(); self._validate_layout()


# ── config / stats ───────────────────────────────────────────────────────────
@dataclass
class PPOGPUConfig:
    total_iterations: int = 1000
    rollout_steps: int = 32
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    critic_lr: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    num_bins: int = ACTION_BINS
    architecture: str = "mlp"
    # Legacy checkpoint/evaluation metadata only; the MLP updater ignores these.
    encoder_depth: int = 2
    save_runtime: bool = False
    hidden: tuple = (512, 512, 512)
    gru_size: int = 256
    recurrent_seq_len: int = 32
    activation: str = "tanh"
    # Library opt-in keeps old comparison fixtures unchanged. New CLI main runs
    # explicitly enable it; comparison commands explicitly disable it.
    aux_pred: bool = False
    aux_coef: float = 0.1
    normalize_obs: bool = True
    norm_adv: bool = True
    seed: int = 0
    device: str = "cuda"
    # ── iteration 스케줄 (sched_period iter 마다 단계 k=(it-1)//period 증가) ──────────
    # lr/entropy = max(floor, base*decay^k). Zero floors preserve legacy callers.
    # The main command opts into halving/floors; comparison schedules stay OFF.
    # Gamma/damage/terminal/aux weights remain fixed; geometry has its own ladder.
    sched_period: int = 2000
    sched_lr_decay: float = 1.0 / 3.0
    sched_ent_decay: float = 1.0 / 3.0
    sched_lr_floor: float = 0.0
    sched_ent_floor: float = 0.0
    sched_rollout_increment: int = 8
    # 0 preserves the legacy unbounded schedule.  Long-running main learners
    # explicitly set a finite cap so one logical iteration cannot grow forever.
    sched_rollout_cap: int = 0
    sched_shaping_ladder: tuple = (1.0, 0.6, 0.32, 0.12, 0.0)
    # ── opponent pool / gated self-play (원본과 동일 규약) ──────────────────────
    pool_evict_cap: int = 4               # evictable(net) snapshot 최대 수
    selfplay_gate_threshold: float = 0.6  # evictable 최소 EMA ≥ 이 값이면 snapshot 추가
    selfplay_ema_alpha: float = 0.1       # 승률 EMA 갱신율
    pool_sample_temp: float = 0.3         # softmax 온도 τ
    pool_uniform_floor: float = 0.5       # 균등 분배 비율 f
    opp_sample: bool = True               # opponent 행동 확률적 샘플 여부
    milestone_period: int = 500           # permanent snapshot(+capacity) + exploiter 주기
    # ── bounded active league / cold archive ─────────────────────────────────
    league_enabled: bool = False
    league_dir: str = ""
    league_active_cap: int = 24
    league_latest_period: int = 20
    league_recent_period: int = 100
    league_ema_half_life_games: float = 512.0
    league_non_latest_cap: float = 0.12
    league_payoff_games: int = 64
    league_admission_games: int = 256
    league_admission_score: float = 0.65
    league_admission_lcb: float = 0.60
    league_payoff_refresh_period: int = 500
    league_redteam_period: int = 1000
    # ── exploiter ──────────────────────────────────────────────────────────────
    exploiter_iters: int = 1000
    exploiter_win_target: float = 0.75
    exploiter_alternate_altitude_hunt: bool = True
    exploiter_alt_hunt_coef: float = 5.0
    # A run-level, explicitly enumerated intervention.  This is intentionally
    # not a replacement for the role-local adaptive scheduler: only the listed
    # milestones are forced to altitude_hunt, after which adaptive selection
    # resumes normally.
    forced_altitude_hunt_iterations: tuple = ()
    # Zero preserves the historical behaviour.  A positive value permits
    # archive/main milestone maintenance after this iteration while preventing
    # creation of new side learners.
    exploiter_until_iteration: int = 0
    # exploiter 를 scratch 대신 main net 파라미터로 초기화(비슷한 타이밍에 죽어 위상이 겹치는
    # 문제 완화). first(기본 500) iter 의 exploiter 는 그 시점 main net 으로, 나머지 모든
    # exploiter 는 rest(기본 1000) iter 시점 main net 으로 초기화한다.
    exploiter_init_iteration_first: int = 500
    exploiter_init_iteration_rest: int = 1000
    exploiter_lr: float = 1e-4
    exploiter_ent_coef: float = 5e-5
    exploiter_clip_coef: float = 0.4

    def __post_init__(self):
        for name in ("sched_lr_floor", "sched_ent_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.sched_rollout_cap) < 0:
            raise ValueError("sched_rollout_cap must be zero or a positive integer")
        if self.sched_rollout_cap and int(self.sched_rollout_cap) < int(self.rollout_steps):
            raise ValueError("sched_rollout_cap must not be below the base rollout")
        if self.sched_period > 0:
            critic_lr = self.lr if self.critic_lr is None else self.critic_lr
            if self.sched_lr_floor > min(self.lr, critic_lr):
                raise ValueError("sched_lr_floor must not exceed either base learning rate")
            if self.sched_ent_floor > self.ent_coef:
                raise ValueError("sched_ent_floor must not exceed base entropy coefficient")
        if self.league_enabled:
            if not self.league_dir:
                raise ValueError("league_dir is required when the active league is enabled")
        forced = tuple(self.forced_altitude_hunt_iterations)
        if (any(type(value) is not int or value <= 0 for value in forced)
                or tuple(sorted(set(forced))) != forced):
            raise ValueError("forced_altitude_hunt_iterations must be sorted unique positive integers")
        if int(self.exploiter_until_iteration) < 0:
            raise ValueError("exploiter_until_iteration must be zero or a positive integer")
            if self.league_active_cap != sum(ACTIVE_LAYOUT.values()):
                raise ValueError("the approved active league layout is exactly 24 policies")
            for name in ("league_latest_period", "league_recent_period",
                         "league_payoff_refresh_period", "league_redteam_period",
                         "league_payoff_games", "league_admission_games"):
                if int(getattr(self, name)) <= 0:
                    raise ValueError(f"{name} must be positive")
            for name in ("league_admission_score", "league_admission_lcb",
                         "league_non_latest_cap"):
                value = float(getattr(self, name))
                if not math.isfinite(value) or not 0.0 < value < 1.0:
                    raise ValueError(f"{name} must be in (0, 1)")


@dataclass
class GPUIterationStats:
    iteration: int
    global_step: int
    mean_return: float
    mean_length: float
    completed_episodes: float
    win_rate: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    explained_variance: float
    steps_per_sec: float
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


# ── trainer ──────────────────────────────────────────────────────────────────
class PPOGPUTrainer:
    def __init__(self, env, config: PPOGPUConfig):
        self.env = env
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        dev = config.device

        self.nenv = env.nenv
        self.nac = env.nac
        self.obs_dim = int(env.OBS_SIZE)
        self.act_dim = 4
        self.min_alt = float(getattr(env, "min_altitude_m", 304.8))
        validate_reward_mode(STANDARD_REWARD, config.exploiter_alt_hunt_coef)
        if not math.isfinite(config.exploiter_win_target) or not 0 <= config.exploiter_win_target <= 1:
            raise ValueError("exploiter_win_target must be in [0, 1]")
        self.exploiter_history = []
        if not math.isfinite(config.aux_coef) or config.aux_coef < 0.0:
            raise ValueError("aux_coef must be finite and non-negative")
        if config.aux_pred:
            if config.architecture != "mlp" or self.obs_dim != 184:
                raise ValueError("future-position auxiliary training requires the 184D MLP environment")
            if (getattr(self.env, "substeps", 6) != 6
                    or not math.isclose(float(self.env.obr.dt), 0.1, abs_tol=1e-12)):
                raise ValueError("future-position horizons are frozen for 10Hz control")
            self.env.obr.enable_aux_capture()
        if config.league_enabled:
            self.training_protocol = ACTIVE_AUX_PROTOCOL if config.aux_pred else TRAINING_PROTOCOL
        else:
            self.training_protocol = AUX_PROTOCOL if config.aux_pred else LEGACY_TRAINING_PROTOCOL
        self._model_kwargs = dict(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                  num_bins=config.num_bins, hidden=tuple(config.hidden),
                                  activation=config.activation, gru_size=config.gru_size,
                                  architecture=config.architecture, encoder_depth=config.encoder_depth)
        if config.aux_pred:
            self._model_kwargs["aux_pred"] = True
        self.grid = make_action_grid(config.num_bins, device=dev)

        self.model = build_actor_critic(**self._model_kwargs).to(dev)
        self._build_optim()
        self.norm = RunningNorm(self.obs_dim, dev) if config.normalize_obs else None
        self.global_step = 0

        # Opponent matchmaking.  New main runs use a disk-backed archive and a
        # fixed 24-policy GPU league; legacy CPU fixtures can still instantiate
        # the small original pool by leaving league_enabled=False.
        self.archive = None
        # Lazily-created, step-only evaluators are independent from the 4096-env
        # PPO environment.  The key is the number of simultaneously mirrored
        # games (2 * the requested games per policy ordering).
        self._league_eval_envs = {}
        if config.league_enabled:
            self.archive = LeagueArchive(config.league_dir)
            self.pool = ActiveLeaguePool(
                self._model_kwargs, dev, active_cap=config.league_active_cap,
                sample=config.opp_sample,
                ema_half_life_games=config.league_ema_half_life_games,
                non_latest_cap=config.league_non_latest_cap, seed=config.seed)
            self.pool.add(self.model, self.norm, role="latest", created_iteration=0)
            self._latest_clone_iteration = 0
            self._recent_archive_ids = []
            self._last_milestone_archive_id = None
            self._heldout_audit_cursor = 0
            self._role_profile_bandit = _empty_role_profile_bandit()
        else:
            self.pool = OpponentPool(self._model_kwargs, dev, evict_cap=config.pool_evict_cap,
                                     sample=config.opp_sample)
            self.pool.add(self.model, self.norm, permanent=False, ema=0.5)
        self.opp_weights = self._pool_weights()
        self.opp_assign = torch.zeros(self.nenv, dtype=torch.long, device=dev)

        # 롤아웃 버퍼(main 기체만; (T,nenv,...)). 행동은 index(long).
        self._alloc_rollout_buffers(config.rollout_steps)
        self.ep_ret = torch.zeros(self.nenv, device=dev)
        self.ep_len = torch.zeros(self.nenv, device=dev)

        # exploiter 초기화 시드 스냅샷 2종(각 해당 iteration 에서 1회 캡처 후 재사용, resume 대비
        # checkpoint 에도 저장/복원). first = exploiter_init_iteration_first(기본 500) 시점 main
        # net → first iter 의 exploiter 전용. rest = exploiter_init_iteration_rest(기본 1000)
        # 시점 main net → 그 외 모든 exploiter.
        self._exp_init_first = None
        self._exp_init_rest = None

        self._reset_env_state()

    def _alloc_rollout_buffers(self, T):
        """(T,nenv,...) 롤아웃 버퍼 (재)할당. 스케줄로 rollout 이 바뀌면 다시 호출한다."""
        T = int(T)
        dev = self.cfg.device
        self.b_obs = torch.zeros(T, self.nenv, self.obs_dim, device=dev)
        self.b_act = torch.zeros(T, self.nenv, self.act_dim, dtype=torch.long, device=dev)
        self.b_logp = torch.zeros(T, self.nenv, device=dev)
        self.b_rew = torch.zeros(T, self.nenv, device=dev)
        self.b_done = torch.zeros(T, self.nenv, device=dev)
        self.b_val = torch.zeros(T, self.nenv, device=dev)
        if self.cfg.aux_pred:
            self.b_aux_features = torch.zeros(T, self.nenv, AUX_FEATURE_DIM,
                                               device=dev, dtype=torch.float64)
            self.b_aux_labels = torch.zeros(T, self.nenv, AUX_DIM, device=dev)
            self.b_aux_mask = torch.zeros(T, self.nenv, 2, device=dev, dtype=torch.bool)

    def _schedule_contract(self):
        """Checkpoint the immutable bases, not already-decayed live cfg values.

        A fresh resume must request this same schedule before loading weights or
        optimizer state. In particular, do not decay a checkpoint's current
        entropy/rollout value for a second time.
        """
        if int(self.cfg.sched_period or 0) <= 0:
            return None
        b = getattr(self, "_sched_base", None)
        if b is None:
            b = {"rollout": int(self.cfg.rollout_steps), "ent": float(self.cfg.ent_coef),
                 "actor_lr": float(self.cfg.lr),
                 "critic_lr": float(self.cfg.lr if self.cfg.critic_lr is None else self.cfg.critic_lr),
                 "shaping": float(self.env.reward_cfg["shaping_reward_scale"])}
        return {"version": "bounded_main_schedule_v2_rollout_cap", "base": dict(b),
                "period": int(self.cfg.sched_period),
                "lr_decay": float(self.cfg.sched_lr_decay),
                "entropy_decay": float(self.cfg.sched_ent_decay),
                "lr_floor": float(self.cfg.sched_lr_floor),
                "entropy_floor": float(self.cfg.sched_ent_floor),
                "rollout_increment": int(self.cfg.sched_rollout_increment),
                "rollout_cap": int(self.cfg.sched_rollout_cap),
                "shaping_ladder": list(self.cfg.sched_shaping_ladder or (1.0,))}

    def _league_contract(self):
        if not self.cfg.league_enabled:
            return None
        c = self.cfg
        return {"protocol": LEAGUE_PROTOCOL, "layout": dict(ACTIVE_LAYOUT),
                "active_cap": c.league_active_cap,
                "latest_period": c.league_latest_period,
                "recent_period": c.league_recent_period,
                "recent_policy": "chronological_periodic_fifo_v1",
                "early_mixture": "latest50_recent_role50_v1",
                "ema_half_life_games": c.league_ema_half_life_games,
                "non_latest_cap": c.league_non_latest_cap,
                "coordinate_contract": "official_base_37p240778_131p869556_direct_msl_d_v1",
                "evaluator": "step_only_full_episode_stratified_mirrored_stochastic_v3",
                "deployment_policy": "packaged_submission_client_stochastic_v1",
                "payoff_evidence": "cumulative_wdl_paired_block_bootstrap_jeffreys_v3",
                "meta_nash": "paired_block_jeffreys_posterior_mean_v2",
                "coverage": "payoff_response_coverage_v1",
                "payoff_games": c.league_payoff_games,
                "admission_games": c.league_admission_games,
                "admission_score": c.league_admission_score,
                "admission_lcb": c.league_admission_lcb,
                "admission_rule": "point_lcb_or_similar_successor_v3",
                "payoff_refresh_period": c.league_payoff_refresh_period,
                "redteam_period": c.league_redteam_period,
                "exploiter_roles": list(EXPLOITER_ROLES),
                "profile_scheduler": PROFILE_SCHEDULER_PROTOCOL,
                "similar_successor": {
                    "family": "role_x_profile_active_incumbent_v1",
                    "screening": "candidate_point_gt_incumbent_point",
                    "confidence": "candidate_lcb95_gt_incumbent_ucb95"}}

    def _apply_schedule(self, it):
        """sched_period iter 마다 단계 k=(it-1)//period 로: max(floor, base*decay^k),
        rollout += increment·k. gamma 등은 불변. base 값은 첫 호출(=학습/재개 시작) 시점의
        cfg 값으로 고정하고 매 iter iteration 번호만으로 결정 → 재개(resume) 안전.
        (checkpoint 는 cfg 를 복원하지 않으므로 cfg 는 항상 CLI base 값 그대로다.)"""
        period = int(getattr(self.cfg, "sched_period", 0) or 0)
        if period <= 0:
            return
        if not hasattr(self, "_sched_base"):
            clr = self.cfg.critic_lr if self.cfg.critic_lr is not None else self.cfg.lr
            self._sched_base = {"rollout": int(self.cfg.rollout_steps),
                                "ent": float(self.cfg.ent_coef),
                                "actor_lr": float(self.cfg.lr),
                                "critic_lr": float(clr),
                                "shaping": float(self.env.reward_cfg["shaping_reward_scale"])}
            self._sched_phase = -1
        b = self._sched_base
        k = (int(it) - 1) // period
        lr_factor = float(self.cfg.sched_lr_decay) ** k
        self.cfg.ent_coef = max(float(self.cfg.sched_ent_floor),
                                b["ent"] * (float(self.cfg.sched_ent_decay) ** k))
        actor_lr = max(float(self.cfg.sched_lr_floor), b["actor_lr"] * lr_factor)
        critic_lr = max(float(self.cfg.sched_lr_floor), b["critic_lr"] * lr_factor)
        for g in self.actor_opt.param_groups:
            g["lr"] = actor_lr
        for g in self.critic_opt.param_groups:
            g["lr"] = critic_lr
        new_T = b["rollout"] + int(self.cfg.sched_rollout_increment) * k
        rollout_cap = int(self.cfg.sched_rollout_cap or 0)
        if rollout_cap > 0:
            new_T = min(new_T, rollout_cap)
        if new_T != int(self.cfg.rollout_steps):
            self.cfg.rollout_steps = new_T
            self._alloc_rollout_buffers(new_T)
        shaping_ladder = tuple(self.cfg.sched_shaping_ladder or (1.0,))
        shaping_mult = float(shaping_ladder[min(k, len(shaping_ladder) - 1)])
        self.env.reward_cfg["shaping_reward_scale"] = b["shaping"] * shaping_mult
        self.env.reward_cfg["own_damage_weight"] = 1.0
        if k != self._sched_phase:
            self._sched_phase = k
            print(f"[gpu-ppo] 스케줄 단계 k={k} (iter {it}): rollout={self.cfg.rollout_steps}, "
                  f"lr={actor_lr:.3e}, critic_lr={critic_lr:.3e}, ent_coef={self.cfg.ent_coef:.3e}, "
                  f"gamma={self.cfg.gamma:.4f}, shaping_mult={shaping_mult}", flush=True)

    def _build_optim(self):
        c = self.cfg
        clr = c.critic_lr if c.critic_lr is not None else c.lr
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=c.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)

    def _sample_opp(self, n):
        """Sample stable IDs; retirees have zero sampling probability."""
        rows = torch.multinomial(self.opp_weights, n, replacement=True)
        return self.pool.ids.index_select(0, rows)

    def _pool_weights(self):
        c = self.cfg
        return self.pool.weights(c.pool_sample_temp, c.pool_uniform_floor)

    def _reset_env_state(self, *, stagger=True, evaluation=False):
        if evaluation and hasattr(self.env, "reset_evaluation"):
            obs = self.env.reset_evaluation()
        else:
            obs = self.env.reset(stagger=bool(stagger))
        self._next_obs = obs[:, 0, :].contiguous()
        self._next_opp_obs = obs[:, 1, :].contiguous()
        self._next_done = torch.zeros(self.nenv, device=self.cfg.device)
        self.opp_assign = self._sample_opp(self.nenv)
        self._actor_h, self._critic_h = self.model.initial_state(
            self.nenv, self.cfg.device)
        for e in self.pool.entries:
            e["actor_state"] = None
        self.ep_ret.zero_(); self.ep_len.zero_()

    def _refresh_weights(self):
        self.pool.refresh_residents(self.opp_assign, self._next_done)
        self.opp_weights = self._pool_weights()
        # No resampling here. collect_rollout changes only completed lanes.

    def _policy_bundle(self):
        norm = None
        if self.norm is not None:
            norm = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                    for k, v in self.norm.state_dict().items()}
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "norm": norm}

    def _load_policy_bundle(self, bundle):
        self.model.load_state_dict({k: torch.as_tensor(v, device=self.cfg.device)
                                    for k, v in bundle["model"].items()})
        if self.norm is not None:
            if bundle.get("norm") is None:
                self.norm = RunningNorm(self.obs_dim, self.cfg.device)
            else:
                self.norm.load_state_dict({k: torch.as_tensor(v, device=self.cfg.device)
                                           for k, v in bundle["norm"].items()})

    def _frozen_policy(self, bundle):
        from cuda_fdm.external_opponents import is_external, frozen_external
        if is_external(bundle):
            return frozen_external(bundle, self.cfg.device)
        net = build_actor_critic(**self._model_kwargs).to(self.cfg.device)
        net.load_state_dict(bundle["model"]); net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        norm = None
        if bundle.get("norm") is not None:
            norm = RunningNorm(self.obs_dim, self.cfg.device)
            norm.load_state_dict({k: torch.as_tensor(v, device=self.cfg.device)
                                  for k, v in bundle["norm"].items()})
        return net, norm, {"state": None}

    def _pool_roster_snapshot(self):
        """Hold the exact in-process resident roster across isolated evaluation.

        `_reset_env_state()` is allowed to garbage-collect retired opponents
        because it starts fresh episodes.  An evaluator or side learner must
        not make that cleanup visible to the paused main, whose saved live
        assignments can still reference those residents.  Keeping object
        references avoids copying up to 24 frozen networks for every payoff
        evaluation; hidden states and per-policy RNG bytes remain covered by
        `_runtime_state()`.
        """
        return {"entries": list(self.pool.entries), "next_id": int(self.pool.next_id)}

    def _restore_pool_roster(self, state):
        self.pool.entries = list(state["entries"])
        self.pool.next_id = int(state["next_id"])
        self.pool._reindex()

    def _league_eval_env(self, minimum_games):
        """Return a small, cached CUDA env with enough lanes per policy ordering."""
        if torch.device(self.cfg.device).type != "cuda":
            raise RuntimeError("the production league evaluator requires CUDA")
        per_order = max(4, ((int(minimum_games) + 3) // 4) * 4)
        total_games = 2 * per_order
        evaluator = self._league_eval_envs.get(total_games)
        if evaluator is None:
            from cuda_fdm.rl_env import GpuDogfightVecEnv
            evaluator = GpuDogfightVecEnv(
                total_games, substeps=int(getattr(self.env, "substeps", 6)),
                seed=int(self.cfg.seed), device=self.cfg.device,
                min_altitude_m=float(getattr(self.env, "min_altitude_m", 304.8)),
                max_engage_time_s=float(getattr(self.env, "max_engage_time_s", 200.0)),
                reward_cfg=dict(getattr(self.env, "reward_cfg", {})),
                reward_mode=STANDARD_REWARD,
                alt_hunt_coef=float(getattr(self.env, "alt_hunt_coef", 5.0)))
            # Completed lanes autoreset internally but are never counted twice.
            # A small pool avoids needless evaluator-only seed construction.
            evaluator.ic_pool_size = min(128, total_games)
            self._league_eval_envs[total_games] = evaluator
        return evaluator, per_order

    @staticmethod
    def _outcome_summary(scores, altitude_losses):
        scores = np.asarray(scores, dtype=np.float64)
        altitude_losses = np.asarray(altitude_losses, dtype=np.bool_)
        wins = int(np.count_nonzero(scores == 1.0))
        losses = int(np.count_nonzero(scores == 0.0))
        draws = int(scores.size - wins - losses)
        return {"score": float(scores.mean()), "games": int(scores.size),
                "wins": wins, "draws": draws, "losses": losses,
                "alt_loss_rate": float(altitude_losses.mean())}

    @torch.inference_mode()
    def _run_clean_paired_evaluation(self, left, right, minimum_games, seed_block):
        """One complete clean episode per lane with submission-aligned sampling.

        The packaged competition entry point constructs ``MLPActionProvider``
        with ``stochastic=True``.  League evaluation therefore samples the same
        categorical policy instead of silently evaluating its greedy argmax.
        A block-derived RNG seed makes the result exactly replayable; the
        caller restores every global/main RNG afterwards.
        """
        eval_seed = (int(self.cfg.seed) * 1_000_003 + int(seed_block) * 97_409
                     + int(minimum_games) * 193) % (2**31 - 1)
        torch.manual_seed(eval_seed)
        if torch.device(self.cfg.device).type == "cuda":
            torch.cuda.manual_seed_all(eval_seed)
        env, per_order = self._league_eval_env(minimum_games)
        external = any(getattr(p[0], 'obs_dim', 184)==214 for p in (left,right))
        if external:
            env.obr.enable_external_observations()
        obs = env.reset_evaluation(paired=True, seed_block=int(seed_block))
        nenv = env.nenv
        rows = torch.arange(nenv, device=self.cfg.device)
        left_slot = torch.cat((
            torch.zeros(per_order, dtype=torch.long, device=self.cfg.device),
            torch.ones(per_order, dtype=torch.long, device=self.cfg.device)))
        right_slot = 1 - left_slot
        left_model, left_norm, _ = left
        right_model, right_norm, _ = right
        left_state = left_model.initial_state(nenv, self.cfg.device)
        right_state = right_model.initial_state(nenv, self.cfg.device)
        episode_start = torch.ones(nenv, device=self.cfg.device)
        finished = torch.zeros(nenv, dtype=torch.bool, device=self.cfg.device)
        scores = torch.full((nenv,), float("nan"), device=self.cfg.device)
        left_alt_loss = torch.zeros(nenv, dtype=torch.bool, device=self.cfg.device)
        right_alt_loss = torch.zeros_like(left_alt_loss)
        max_steps = math.ceil(float(env.max_engage_time_s) / float(env.obr.dt)) + 4
        final_step = 0
        for step in range(1, max_steps + 1):
            left_obs = obs[rows, left_slot]
            right_obs = obs[rows, right_slot]
            if external:
                full = env.obr.build_external_obs(env.sim.states).view(nenv,2,214)
                if getattr(left_model, 'obs_dim', 184)==214: left_obs = full[rows,left_slot]
                if getattr(right_model, 'obs_dim', 184)==214: right_obs = full[rows,right_slot]
            if left_norm is not None:
                left_obs = left_norm.normalize(left_obs)
            if right_norm is not None:
                right_obs = right_norm.normalize(right_obs)
            left_action, left_state = left_model.act(
                left_obs, state=left_state, episode_start=episode_start, sample=True)
            right_action, right_state = right_model.act(
                right_obs, state=right_state, episode_start=episode_start, sample=True)
            controls = torch.empty(nenv, 2, self.act_dim, device=self.cfg.device)
            controls[rows, left_slot] = action_to_env(
                left_action, self.cfg.num_bins, self.grid)
            controls[rows, right_slot] = action_to_env(
                right_action, self.cfg.num_bins, self.grid)
            obs, reward, done, info = env.step(controls)
            active = ~finished
            hp = info["terminal_hp"]
            altitude = info["terminal_alt_m"]
            # Fixed-size per-lane reductions avoid CUDA nonzero/indexing syncs.
            finite_lanes = (info["terminal_state_finite"]
                            & torch.isfinite(hp).reshape(nenv, -1).all(dim=1)
                            & torch.isfinite(altitude).reshape(nenv, -1).all(dim=1)
                            & torch.isfinite(reward).reshape(nenv, -1).all(dim=1))
            finite = (finite_lanes | ~active).all()
            newly = done.bool() & active
            left_hp = hp[rows, left_slot]
            right_hp = hp[rows, right_slot]
            left_alt = altitude[rows, left_slot]
            right_alt = altitude[rows, right_slot]
            left_dead = (left_hp <= 0.0) | (left_alt < env.min_altitude_m)
            right_dead = (right_hp <= 0.0) | (right_alt < env.min_altitude_m)
            both_alive = ~left_dead & ~right_dead
            win = ((right_dead & ~left_dead)
                   | (both_alive & (left_hp > right_hp + 1e-9)))
            loss = ((left_dead & ~right_dead)
                    | (both_alive & (left_hp < right_hp - 1e-9)))
            outcome = win.float() + 0.5 * (~win & ~loss).float()
            scores = torch.where(newly, outcome, scores)
            left_alt_loss = torch.where(newly, left_alt < env.min_altitude_m, left_alt_loss)
            right_alt_loss = torch.where(newly, right_alt < env.min_altitude_m, right_alt_loss)
            finished |= newly
            episode_start = done.float()
            final_step = step
            finite_ok, complete = torch.stack((finite, finished.all())).tolist()
            if not finite_ok:
                raise FloatingPointError("non-finite clean league evaluation trajectory")
            if complete:
                break
        if not bool(finished.all()) or not bool(torch.isfinite(scores).all()):
            raise RuntimeError("clean league evaluation did not finish exactly one game per lane")

        return self._summarize_clean_pair(scores, left_alt_loss, right_alt_loss,
                                          per_order, seed_block, eval_seed, final_step)

    def _summarize_clean_pair(self, scores, left_alt_loss, right_alt_loss,
                              per_order, seed_block, eval_seed, final_step):
        # `left_slot` tracks the policy argument, not physical plane index, so
        # every value here is already from the left policy's perspective even
        # in the swapped half of the bank.
        left_policy_scores = scores.detach().cpu().numpy()
        left_policy_crashes = left_alt_loss.detach().cpu().numpy()
        right_policy_crashes = right_alt_loss.detach().cpu().numpy()
        forward_scores = left_policy_scores[:per_order]
        swapped_left_policy_scores = left_policy_scores[per_order:]
        # Historical `reverse` is expressed from the right policy's perspective;
        # the top-level record remains the left policy over both role orderings.
        reverse_scores = 1.0 - swapped_left_policy_scores
        forward = self._outcome_summary(
            forward_scores, left_policy_crashes[:per_order])
        reverse = self._outcome_summary(
            reverse_scores, right_policy_crashes[per_order:])
        aggregate = self._outcome_summary(
            left_policy_scores, left_policy_crashes)
        opponent_altitude_losses = int(np.count_nonzero(right_policy_crashes))
        wins_by_opponent_altitude = int(np.count_nonzero(
            (left_policy_scores == 1.0) & right_policy_crashes))
        own_altitude_losses = int(np.count_nonzero(left_policy_crashes))
        losses_by_own_altitude = int(np.count_nonzero(
            (left_policy_scores == 0.0) & left_policy_crashes))
        aggregate.update({
            # Top-level diagnostics are always expressed from the ``left``
            # policy's perspective across both mirrored orderings.
            "opponent_alt_loss_rate": float(right_policy_crashes.mean()),
            "opponent_alt_loss_count": opponent_altitude_losses,
            "wins_by_opponent_altitude": wins_by_opponent_altitude,
            "wins_by_opponent_altitude_rate": float(
                wins_by_opponent_altitude / left_policy_scores.size),
            "wins_by_opponent_altitude_share": float(
                wins_by_opponent_altitude / aggregate["wins"])
                if aggregate["wins"] else 0.0,
            "own_alt_loss_count": own_altitude_losses,
            "losses_by_own_altitude": losses_by_own_altitude,
            "losses_by_own_altitude_rate": float(
                losses_by_own_altitude / left_policy_scores.size),
            "altitude_diagnostic_protocol":
                "mirrored_terminal_cause_observability_v2",
        })
        low, high = paired_score_interval(
            forward_scores, swapped_left_policy_scores,
            seed=int(self.cfg.seed) + int(seed_block) * 1_000_003 + per_order)
        aggregate.update(lcb95=low, ucb95=high, forward=forward, reverse=reverse,
                         paired_blocks=per_order, seed_block=int(seed_block),
                         paired_scores=(0.5 * (
                             forward_scores + swapped_left_policy_scores)).tolist(),
                         deterministic=False, stochastic=True,
                         policy_rng_seed=int(eval_seed), steps=int(final_step))
        return aggregate

    def _evaluate_pair(self, left_bundle, right_bundle, minimum_games, paired=True,
                       seed_block=0):
        """Isolated replayable stochastic evaluator; PPO buffers are unused."""
        if not paired:
            raise ValueError("active-league evaluation requires mirrored policy orderings")
        started = time.perf_counter()
        saved_runtime = self._runtime_state()
        snapshot_done = time.perf_counter()
        policies_done = snapshot_done
        evaluation_done = snapshot_done
        succeeded = False
        try:
            left = self._frozen_policy(left_bundle)
            right = self._frozen_policy(right_bundle)
            policies_done = time.perf_counter()
            result = self._run_clean_paired_evaluation(
                left, right, minimum_games, seed_block)
            evaluation_done = time.perf_counter()
            succeeded = True
            return result
        finally:
            # Model construction and CUDA kernels may consume global RNG.  The
            # paused main environment, opponent hidden/RNG and all trainer RNGs
            # are restored exactly before training resumes.
            self._restore_runtime(saved_runtime)
            if succeeded:
                restored = time.perf_counter()
                print(f"[league-eval-perf] main={getattr(self, 'iteration', 0)} "
                      f"blocks={minimum_games} seed={seed_block} "
                      f"snapshot_s={snapshot_done-started:.4f} "
                      f"policy_s={policies_done-snapshot_done:.4f} "
                      f"evaluation_s={evaluation_done-policies_done:.4f} "
                      f"restore_s={restored-evaluation_done:.4f} "
                      f"total_s={restored-started:.4f}", flush=True)

    def _archive_current(self, kind, *, profile="standard", admitted=True,
                         payoff_eligible=True, metrics=None):
        if self.archive is None:
            raise RuntimeError("archive is disabled")
        bundle = self._policy_bundle()
        return self.archive.add(bundle["model"], bundle["norm"], kind=kind,
                                iteration=int(getattr(self, "iteration", 0)),
                                profile=profile, admitted=admitted,
                                payoff_eligible=payoff_eligible, metrics=metrics)

    def _archive_bundle(self, bundle, kind, *, profile="standard", admitted=True,
                        payoff_eligible=True, metrics=None):
        return self.archive.add(bundle["model"], bundle.get("norm"), kind=kind,
                                iteration=int(getattr(self, "iteration", 0)),
                                profile=profile, admitted=admitted,
                                payoff_eligible=payoff_eligible, metrics=metrics)

    def _sync_active_scores_to_archive(self):
        if self.archive is None:
            return
        for entry in self.pool.active_entries():
            archive_id = entry.get("archive_id")
            if archive_id is None or archive_id not in self.archive.records:
                continue
            record = self.archive.records[archive_id]
            record["current_score"] = float(entry["ema"])
            record["past_best"] = max(float(record.get("past_best", 0.5)),
                                      float(entry.get("past_best", entry["ema"])))
            record["regression"] = max(0.0, record["past_best"] - record["current_score"])

    def _record_payoff_result(self, left_id, right_id, result, *, seed_block=None):
        wins = float(result["wins"]); draws = float(result["draws"])
        losses = float(result["losses"])
        self.archive.accumulate_payoff(left_id, right_id, wins=wins, draws=draws,
                                       losses=losses, seed_block=seed_block,
                                       lcb95=result.get("lcb95"),
                                       ucb95=result.get("ucb95"),
                                       paired_blocks=result.get("paired_blocks"),
                                       paired_scores=result.get("paired_scores"))

    def _refresh_payoff_row(self, archive_id, *, seed_block=0, force_ids=()):
        if self.archive is None:
            return
        left = self.archive.load_policy(archive_id)
        force_ids = {int(x) for x in force_ids}
        parallel = None
        workers = int(os.environ.get("AIP_LEAGUE_EVAL_WORKERS", "1"))
        batch_size = int(os.environ.get("AIP_LEAGUE_BATCH_SIZE", "0"))
        if batch_size > 1 or workers > 1:
            missing = [other for other in self.archive.strategic_ids(include_heldout=False)
                       if other != archive_id and (
                           str(other) not in self.archive.payoff.get(str(archive_id), {})
                           or other in force_ids)]
            # Shared physics has no subprocess startup. Keep singleton work serial.
            if (batch_size > 1 and len(missing) >= 2) or (batch_size <= 1 and len(missing) >= 24):
                started = time.perf_counter()
                try:
                    if batch_size > 1:
                        from cuda_fdm.batched_payoff import evaluate_row
                        completed = evaluate_row(self, left, missing,
                                                 seed_block=seed_block, capacity=batch_size)
                    else:
                        from cuda_fdm.parallel_payoff import evaluate_row
                        completed = evaluate_row(self, left, missing,
                                                 seed_block=seed_block, workers=workers)
                    if set(completed) != set(missing):
                        raise RuntimeError("incomplete payoff row; refusing partial commit")
                    parallel = completed
                    print(f"[league-row-perf] archive={archive_id} edges={len(missing)} "
                          f"batch_size={batch_size} workers={workers} "
                          f"seconds={time.perf_counter()-started:.2f}", flush=True)
                except Exception as exc:
                    # No payoff or roster mutation happened; replay original path.
                    print(f"[league-row-perf] parallel failed; sequential fallback: {exc!r}",
                          flush=True)
        for other in self.archive.strategic_ids(include_heldout=False):
            if other == archive_id:
                if str(other) not in self.archive.payoff.get(str(archive_id), {}):
                    self.archive.set_payoff(archive_id, archive_id, 0.5,
                                            self.cfg.league_payoff_games)
                continue
            if (str(other) in self.archive.payoff.get(str(archive_id), {})
                    and other not in force_ids):
                continue
            result = (parallel[other] if parallel is not None else
                      self._evaluate_pair(left, self.archive.load_policy(other),
                                          self.cfg.league_payoff_games, paired=True,
                                          seed_block=seed_block))
            self._record_payoff_result(archive_id, other, result, seed_block=seed_block)
        self.archive.persist()

    def _refresh_uncertain_payoff_edge(self, *, seed_block):
        """Refresh one bounded, influential old edge instead of O(N^2) sweeps."""
        if self.archive is None:
            return None
        identities = self.archive.strategic_ids(include_heldout=False)
        current = self._last_milestone_archive_id
        pairs = [(left, right) for index, left in enumerate(identities)
                 for right in identities[index + 1:]
                 if current not in (left, right)]
        if not pairs:
            return None
        def priority(pair):
            left, right = pair
            importance = (0.10 + float(self.archive.records[left].get("nash_mass", 0.0))
                          + float(self.archive.records[right].get("nash_mass", 0.0)))
            return self.archive.payoff_uncertainty(left, right) * importance
        left, right = max(pairs, key=lambda pair: (priority(pair), -pair[0], -pair[1]))
        result = self._evaluate_pair(self.archive.load_policy(left),
                                     self.archive.load_policy(right),
                                     self.cfg.league_payoff_games, paired=True,
                                     seed_block=seed_block)
        self._record_payoff_result(left, right, result, seed_block=seed_block)
        self.archive.persist()
        return left, right

    def _refresh_active_core(self):
        if self.archive is None:
            return
        self._sync_active_scores_to_archive()
        self.archive.refresh_meta(self._last_milestone_archive_id)
        active_before = {e.get('archive_id') for e in self.pool.active_entries()
                         if e.get('role') in ('core','challenger')}
        eligible = {i for i,r in self.archive.records.items()
                    if i in active_before or not r.get('metrics',{}).get('recovery_required',False)}
        challengers = [identity for identity, record in sorted(self.archive.records.items(), reverse=True)
                       if identity in eligible and record.get("admitted") and record.get("kind", "").startswith("exploiter")]
        challengers.sort(key=lambda i:(self.archive.records[i].get('metrics',{}).get('activation_iteration',
                               self.archive.records[i]['iteration']), i), reverse=True)
        challengers = list(reversed(challengers[:ACTIVE_LAYOUT["challenger"]]))
        recent = [e.get("archive_id") for e in self.pool.active_entries()
                  if e.get("role") == "recent" and e.get("archive_id") is not None]
        core, coverage = self.archive.select_core(
            self._last_milestone_archive_id, limit=ACTIVE_LAYOUT["core"],
            exclude=recent + challengers, eligible_ids=eligible)
        self.pool.sync_archive_roles(self.archive, core, challengers, coverage)
        for identity in active_before - set(core) - set(challengers):
            if identity in self.archive.records:
                self.archive.records[identity]['metrics']['recovery_required']=True
        for entry in self.pool.active_entries():
            archive_id = entry.get("archive_id")
            if archive_id in self.archive.records:
                record = self.archive.records[archive_id]
                entry["nash_mass"] = float(record.get("nash_mass", 0.0))
                entry["ema"] = float(record.get("current_score", entry["ema"]))
                entry["past_best"] = float(record.get("past_best", entry["past_best"]))
        self.archive.persist()

    @torch.no_grad()
    def _action_kl_to_entry(self, entry, maximum_states=1024):
        count = min(int(maximum_states), self.nenv)
        obs = self._next_obs[:count]
        left = self.norm.normalize(obs) if self.norm is not None else obs
        right = entry["norm"].normalize(obs) if entry["norm"] is not None else obs
        a = self.model.actor_logits(left).view(-1, self.act_dim, self.cfg.num_bins)
        b = entry["net"].actor_logits(right).view(-1, self.act_dim, self.cfg.num_bins)
        pa = torch.softmax(a.float(), -1)
        value = (pa * (torch.log_softmax(a.float(), -1) - torch.log_softmax(b.float(), -1))).sum(-1)
        return float(value.mean())

    def _maybe_update_latest_and_recent(self, iteration):
        if self.archive is None:
            return None
        event = None
        latest = next((e for e in self.pool.active_entries()
                       if e.get("role") == "latest"), None)
        latest_lag = iteration - self._latest_clone_iteration
        latest_kl = (float("inf") if latest is None
                     else self._action_kl_to_entry(latest))
        if latest_kl >= 0.02 or latest_lag >= self.cfg.league_latest_period:
            self.pool.replace_latest(self.model, self.norm, iteration)
            self._latest_clone_iteration = iteration
            event = "latest"
        if iteration % self.cfg.league_recent_period:
            return event
        # Recent is a chronological curriculum ring, not a mastery gate.  The
        # latest clone already provides drift-sensitive self-play, so computing
        # a second KL here only to pass the periodic lag condition is redundant.
        archive_id = self._archive_current(
            "recent_main", admitted=True, payoff_eligible=False,
            metrics={"selection": "chronological_periodic_fifo_v1",
                     "period": int(self.cfg.league_recent_period)})
        before_ids = {e.get('archive_id') for e in self.pool.active_entries()}
        self.pool.add_recent_bundle(self.model, self.norm, archive_id, iteration)
        after_ids = {e.get('archive_id') for e in self.pool.active_entries()}
        for identity in before_ids - after_ids:
            if identity in self.archive.records:
                self.archive.records[identity]['metrics']['recovery_required'] = True
        self._recent_archive_ids = [e["archive_id"] for e in self.pool.active_entries()
                                    if e.get("role") == "recent"]
        event = "recent"
        return event

    def _audit_heldout_redteam(self, iteration):
        if self.archive is None or iteration % self.cfg.league_redteam_period:
            return []
        active = {e.get('archive_id') for e in self.pool.active_entries()}
        heldout = [identity for identity, record in sorted(self.archive.records.items())
                   if identity not in active and identity != self._last_milestone_archive_id]
        if not heldout:
            return []
        start = int(self._heldout_audit_cursor) % len(heldout)
        count = min(8, len(heldout))
        selected = [heldout[(start + offset) % len(heldout)] for offset in range(count)]
        self._heldout_audit_cursor = (start + count) % len(heldout)
        promoted = []
        current = self._policy_bundle()
        audit_block = iteration // max(1, self.cfg.league_redteam_period)
        for identity in selected:
            result = self._evaluate_pair(current, self.archive.load_policy(identity),
                                         self.cfg.league_payoff_games, paired=True,
                                         seed_block=audit_block)
            milestone_record = (self.archive.records.get(self._last_milestone_archive_id)
                                if self._last_milestone_archive_id is not None else None)
            if milestone_record is not None and int(milestone_record["iteration"]) == int(iteration):
                self._record_payoff_result(self._last_milestone_archive_id, identity, result,
                                           seed_block=audit_block)
            record = self.archive.records[identity]
            record["metrics"].update({
                "last_audit_iteration": int(iteration),
                "last_audit_seed_block": int(audit_block),
                "last_audit_main_score": float(result["score"]),
                "last_audit_main_lcb95": float(result["lcb95"]),
                "last_audit_main_ucb95": float(result["ucb95"]),
                "last_audit_games": int(result["games"]),
                "last_audit_wins": int(result["wins"]),
                "last_audit_draws": int(result["draws"]),
                "last_audit_losses": int(result["losses"]),
            })
            from cuda_fdm.league import recovery_passes
            record['metrics']['recovery_required'] = not recovery_passes(result)
            record['metrics']['recovery_main_ucb_threshold'] = 0.70
            if recovery_passes(result):
                record["admitted"] = True
                record['payoff_eligible'] = True
                record['metrics'].setdefault('original_kind',record['kind'])
                record['metrics']['activation_iteration'] = int(iteration)
                record["kind"] = "exploiter_redteam_promoted"
                record["metrics"]["promotion_main_score"] = result["score"]
                record["metrics"]["promotion_main_ucb95"] = result["ucb95"]
                record["metrics"]["promotion_games"] = result["games"]
                self._refresh_payoff_row(identity, seed_block=audit_block)
                promoted.append(identity)
            print(f"[archive-recovery] it={iteration} id={identity} main_score={result['score']:.4f} "
                  f"main_ucb95={result['ucb95']:.4f} threshold=0.70 pass={recovery_passes(result)}",flush=True)
        self.archive.persist()
        return promoted

    # ── 롤아웃 수집 (main=학습, opponent=frozen) ──────────────────────────────
    @torch.no_grad()
    def collect_rollout(self, opp_kind="pool", frozen_opp=None, update_norm=True):
        cfg = self.cfg
        T = cfg.rollout_steps
        gamma = cfg.gamma
        dev = cfg.device
        P = self.pool.resident_size()
        opponent_ids = self.pool.ids.clone()
        ret_sum = torch.zeros((), device=dev)
        len_sum = torch.zeros((), device=dev)
        ep_count = torch.zeros((), device=dev)
        win_sum = torch.zeros((), device=dev)
        loss_sum = torch.zeros((), device=dev)
        alt_loss_sum = torch.zeros((), device=dev)   # main 이 고도제한 위반으로 패한 게임 수
        opponent_alt_loss_sum = torch.zeros((), device=dev)
        win_by_opp = torch.zeros(P, device=dev)
        loss_by_opp = torch.zeros(P, device=dev)
        ep_by_opp = torch.zeros(P, device=dev)       # opponent 별 완료 에피소드 수(무승부 포함, EMA 분모)
        trajectory_finite = torch.ones((), dtype=torch.bool, device=dev)

        for t in range(T):
            if cfg.aux_pred:
                # Features correspond to s_t / _next_obs, including autoreset
                # lanes. The existing observation kernel already computed them.
                self.b_aux_features[t].copy_(self.env.obr.aux_features)
            if opp_kind == "pool":
                sampled = self._sample_opp(self.nenv)
                self.opp_assign = torch.where(self._next_done.bool(), sampled, self.opp_assign)

            main_obs = self._next_obs
            if self.norm is not None:
                if update_norm:
                    self.norm.update(main_obs)
                obs_n = self.norm.normalize(main_obs)
            else:
                obs_n = main_obs
            self.b_obs[t] = obs_n
            self.b_done[t] = self._next_done

            state = (self._actor_h, self._critic_h)
            act_idx, logp, _, state = self.model.actor_step(
                obs_n, state=state, episode_start=self._next_done)
            value, state = self.model.value_step(
                obs_n, state=state, episode_start=self._next_done)
            self._actor_h, self._critic_h = tuple(x.detach() for x in state)
            self.b_act[t] = act_idx.long()
            self.b_logp[t] = logp
            self.b_val[t] = value

            if opp_kind == "pool":
                external_obs = None
                if any(getattr(e['net'], 'obs_dim', 184)==214 for e in self.pool.entries):
                    external_obs = self.env.obr.build_external_obs(self.env.sim.states).view(self.nenv,2,214)[:,1]
                extra = {'external_obs':external_obs} if external_obs is not None else {}
                opp_idx = self.pool.act(
                    self._next_opp_obs, self.opp_assign, self._next_done, **extra)
            else:
                oo = self._next_opp_obs
                on = frozen_opp[1].normalize(oo) if frozen_opp[1] is not None else oo
                holder = frozen_opp[2]
                if holder.get("state") is None:
                    holder["state"] = frozen_opp[0].initial_state(self.nenv, dev)
                opp_idx, frozen_state = frozen_opp[0].act(
                    on, state=holder["state"], episode_start=self._next_done,
                    sample=cfg.opp_sample)
                holder["state"] = tuple(x.detach() for x in frozen_state)

            act = torch.empty(self.nac, self.act_dim, device=dev)
            act[0::2] = action_to_env(act_idx, cfg.num_bins, self.grid)
            act[1::2] = action_to_env(opp_idx, cfg.num_bins, self.grid)
            obs, reward, done, info = self.env.step(act)
            trajectory_finite.logical_and_(info["terminal_state_finite"].all())

            raw_reward = reward[:, 0].float()
            done_env = done.float()

            # A 200-second timeout settles the match using HP; no reward exists
            # beyond that task terminal. Keep raw damage/geometry/safety/result
            # reward intact. GAE's done mask below zeros the future value for all
            # completed matches, but still bootstraps live rollout boundaries.
            self.b_rew[t] = raw_reward

            self._next_obs = obs[:, 0, :]
            self._next_opp_obs = obs[:, 1, :]
            self._next_done = done_env
            self.global_step += self.nenv

            self.ep_ret += raw_reward
            self.ep_len += 1.0
            ret_sum += (self.ep_ret * done_env).sum()
            len_sum += (self.ep_len * done_env).sum()
            ep_count += done_env.sum()
            keep = 1.0 - done_env
            self.ep_ret = self.ep_ret * keep
            self.ep_len = self.ep_len * keep

            th = info["terminal_hp"]; ta = info["terminal_alt_m"]
            own_alt_dead = ta[:, 0] < self.min_alt
            own_dead = (th[:, 0] <= 0.0) | own_alt_dead
            opp_dead = (th[:, 1] <= 0.0) | (ta[:, 1] < self.min_alt)
            both_alive = (~own_dead) & (~opp_dead)
            win = (opp_dead & ~own_dead) | (both_alive & (th[:, 0] > th[:, 1] + 1e-9))
            loss = (own_dead & ~opp_dead) | (both_alive & (th[:, 0] < th[:, 1] - 1e-9))
            done_b = done.bool()
            dwin = (win & done_b).float()
            dloss = (loss & done_b).float()
            win_sum += dwin.sum(); loss_sum += dloss.sum()
            # 고도제한 패배: 패로 판정된 게임 중 main 이 min_alt 아래로 내려간 경우.
            alt_loss_sum += (loss & own_alt_dead & done_b).float().sum()
            # Diagnostic only: count completed games in which the opponent
            # crossed the hard deck and the learner won.  ``ta`` is already
            # returned by the environment for both aircraft, so this is valid
            # for frozen and resident-pool opponents and adds no policy
            # inference, host sync, reward, advantage, EMA or admission input.
            opponent_alt_loss_sum += (
                win & (ta[:, 1] < self.min_alt) & done_b
            ).float().sum()
            if opp_kind == "pool":       # per-opponent 집계(EMA 갱신용, sync-free scatter).
                rows = self.pool.rows_for_ids(self.opp_assign)
                win_by_opp.scatter_add_(0, rows, dwin)
                loss_by_opp.scatter_add_(0, rows, dloss)
                ep_by_opp.scatter_add_(0, rows, done_b.float())

        last_n = self.norm.normalize(self._next_obs) if self.norm is not None else self._next_obs
        last_value, _ = self.model.value_step(
            last_n, state=(self._actor_h, self._critic_h),
            episode_start=self._next_done)

        adv = torch.zeros_like(self.b_rew)
        lastgae = torch.zeros(self.nenv, device=dev)
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - self._next_done
                next_values = last_value
            else:
                next_nonterminal = 1.0 - self.b_done[t + 1]
                next_values = self.b_val[t + 1]
            delta = self.b_rew[t] + gamma * next_values * next_nonterminal - self.b_val[t]
            lastgae = delta + gamma * cfg.gae_lambda * next_nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + self.b_val
        if cfg.aux_pred:
            labels, mask = build_future_labels(self.b_aux_features, self.b_done, self.env.obr.dt)
            self.b_aux_labels.copy_(labels)
            self.b_aux_mask.copy_(mask)

        # Exactly one aggregate host check per rollout, before any gradient update.
        # NaN physics may otherwise be sanitized into finite observations and draws.
        finite_checks = [trajectory_finite]
        finite_checks.extend(torch.isfinite(x).all() for x in
                             (self.b_obs, self.b_rew, self.b_val, self.b_logp, adv, ret))
        if cfg.aux_pred:
            finite_checks.extend(torch.isfinite(x).all() for x in
                                 (self.b_aux_features, self.b_aux_labels))
        if not bool(torch.stack(finite_checks).all()):
            raise FloatingPointError("non-finite physical trajectory or PPO rollout; update refused")

        stats = {"ret_sum": ret_sum, "len_sum": len_sum, "ep_count": ep_count,
                 "win_sum": win_sum, "loss_sum": loss_sum, "alt_loss_sum": alt_loss_sum,
                 "opponent_alt_loss_sum": opponent_alt_loss_sum,
                 "opponent_ids": opponent_ids,
                 "win_by_opp": win_by_opp, "loss_by_opp": loss_by_opp, "ep_by_opp": ep_by_opp}
        return adv, ret, stats

    # ── MLP 전용 PPO: GPU flat transition minibatches, no TBPTT/padding ──────
    def update(self, adv, ret):
        """Train on every valid transition exactly once per epoch.

        GAE is already computed with episode/timeout masks in collect_rollout.
        MLP needs no episode split, hidden-state buffer or padded GPU tensors.
        Shuffling individual transitions changes minibatch membership relative
        to historical sequence shuffling; this is explicitly protocol v4, not
        a bitwise-equivalent continuation of the old architecture experiment.
        """
        if self.model.is_recurrent:
            raise ValueError("CUDA PPO updater is MLP-only; GRU is inference/evaluation-only")
        cfg = self.cfg
        obs = self.b_obs.flatten(0, 1)
        act = self.b_act.flatten(0, 1)
        old_logp = self.b_logp.flatten()
        if cfg.aux_pred:
            aux_labels = self.b_aux_labels.flatten(0, 1)
            aux_mask = self.b_aux_mask.flatten(0, 1)
            actor_sse = torch.zeros(2, device=cfg.device)
            critic_sse = torch.zeros_like(actor_sse)
            aux_count = torch.zeros_like(actor_sse)
        advantages, returns = adv.flatten(), ret.flatten()
        count = int(obs.shape[0])
        if count == 0 or cfg.update_epochs < 1 or cfg.num_minibatches < 1:
            raise ValueError("MLP PPO requires nonempty rollout, positive epochs and minibatches")
        if tuple(adv.shape) != tuple(self.b_val.shape) or tuple(ret.shape) != tuple(self.b_val.shape):
            raise ValueError("advantage/return shape must match the rollout values")
        mb_size = (count + cfg.num_minibatches - 1) // cfg.num_minibatches
        clip = cfg.clip_coef
        last_pl = last_vl = last_ent = last_kl = last_cf = torch.zeros((), device=cfg.device)
        last_actor_grad = last_critic_grad = torch.zeros((), device=cfg.device)
        early = False
        epoch = 0
        optimizer_steps = 0
        for epoch in range(cfg.update_epochs):
            perm = torch.randperm(count, device=cfg.device)
            kls = []
            for s in range(0, count, mb_size):
                mb = perm[s:s + mb_size]
                mb_obs = obs.index_select(0, mb)
                if cfg.aux_pred:
                    mb_labels = aux_labels.index_select(0, mb)
                    mb_mask = aux_mask.index_select(0, mb)
                    new_logp, entropy, prediction = self.model.evaluate_actions_with_aux(
                        mb_obs, act.index_select(0, mb))
                    actor_aux, a_sse, a_count = auxiliary_error(prediction, mb_labels, mb_mask)
                    actor_sse += a_sse.detach()
                    aux_count += a_count.detach()
                else:
                    new_logp, entropy = self.model.evaluate_actions(
                        mb_obs, act.index_select(0, mb))
                log_ratio = new_logp - old_logp.index_select(0, mb)
                ratio = log_ratio.exp()

                mb_adv = advantages.index_select(0, mb)
                # Keep sample std for ordinary minibatches; a singleton has
                # no meaningful normalization and must not produce NaN.
                if cfg.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg1, pg2).mean()
                ent = entropy.mean()

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss = policy_loss - cfg.ent_coef * ent
                if cfg.aux_pred:
                    actor_loss = actor_loss + cfg.aux_coef * actor_aux
                actor_loss.backward()
                last_actor_grad = nn.utils.clip_grad_norm_(self.model.actor_parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                if cfg.aux_pred:
                    new_value, prediction = self.model.value_with_aux(mb_obs)
                    critic_aux, c_sse, _ = auxiliary_error(prediction, mb_labels, mb_mask)
                    critic_sse += c_sse.detach()
                else:
                    new_value = self.model.get_value(mb_obs)
                value_loss = 0.5 * ((new_value - returns.index_select(0, mb)) ** 2).mean()
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss = cfg.vf_coef * value_loss
                if cfg.aux_pred:
                    critic_loss = critic_loss + cfg.aux_coef * critic_aux
                critic_loss.backward()
                last_critic_grad = nn.utils.clip_grad_norm_(self.model.critic_parameters(), cfg.max_grad_norm)
                self.critic_opt.step()
                optimizer_steps += 1

                with torch.no_grad():
                    kls.append(((ratio - 1.0) - log_ratio).mean())
                    clipfrac = ((ratio - 1.0).abs() > clip).float().mean()
                last_pl = policy_loss.detach(); last_vl = value_loss.detach()
                last_ent = ent.detach(); last_cf = clipfrac
            last_kl = torch.stack(kls).mean()
            # 기존과 같은 epoch 끝 판단 시점만 CPU scalar로 확인한다.
            if cfg.target_kl is not None and float(last_kl) > cfg.target_kl:
                early = True
                break

        b_ret = ret.reshape(-1)
        b_val = self.b_val.reshape(-1)
        var_y = b_ret.var() if count > 1 else torch.zeros((), device=cfg.device)
        ev = torch.where(var_y == 0, torch.zeros((), device=cfg.device),
                         1.0 - (b_ret - b_val).var(correction=1 if count > 1 else 0) / (var_y + 1e-8))
        result = {"pl": last_pl, "vl": last_vl, "ent": last_ent, "kl": last_kl,
                "cf": last_cf, "ev": ev, "epochs": epoch + 1, "early": early,
                "actor_grad_norm": last_actor_grad.detach(),
                "critic_grad_norm": last_critic_grad.detach(),
                "optimizer_steps": optimizer_steps}
        if cfg.aux_pred:
            # Epoch/minibatch aggregate diagnostics stay on GPU until the normal
            # iteration logging boundary. Baseline predicts zero CV residual.
            baseline_sse = (aux_labels.view(-1, 2, 3).square().sum(-1) * aux_mask).sum(0)
            baseline_count = aux_mask.sum(0).float() * 3.0
            result["aux_actor_mse"] = actor_sse.sum() / aux_count.sum().clamp_min(1.)
            result["aux_critic_mse"] = critic_sse.sum() / aux_count.sum().clamp_min(1.)
            result["aux_baseline_mse"] = baseline_sse.sum() / baseline_count.sum().clamp_min(1.)
            for slot, name in enumerate(("opp", "self")):
                result[f"aux_{name}_coverage"] = aux_mask[:, slot].float().mean()
                for owner, sse, counts in (("actor", actor_sse, aux_count),
                                            ("critic", critic_sse, aux_count),
                                            ("baseline", baseline_sse, baseline_count)):
                    result[f"aux_{owner}_{name}_rmse_m"] = (
                        sse[slot] / counts[slot].clamp_min(1.)).sqrt() * AUX_POS_SCALE_M
        return result

    # ── main 상태 저장/복원 (exploiter 학습이 self.model 등을 임시 사용) ────────
    def _snapshot_learner(self):
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "actor_opt": copy.deepcopy(self.actor_opt.state_dict()),
                "critic_opt": copy.deepcopy(self.critic_opt.state_dict()),
                "norm": (self.norm.state_dict() if self.norm is not None else None),
                "global_step": int(self.global_step)}

    def _capture_exploiter_init(self):
        """현재 main net(+norm) 을 CPU 로 복사해 exploiter 초기화 시드로 보관.
        norm 은 live 버퍼 참조가 아니라 clone 을 저장(이후 학습에 오염되지 않도록)."""
        norm_sd = None
        if self.norm is not None:
            norm_sd = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                       for k, v in self.norm.state_dict().items()}
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "norm": norm_sd,
                "iteration": int(getattr(self, "iteration", 0))}

    def _restore_learner(self, snap):
        self.model.load_state_dict({k: v.to(self.cfg.device) for k, v in snap["model"].items()})
        self._build_optim()
        self.actor_opt.load_state_dict(snap["actor_opt"])
        self.critic_opt.load_state_dict(snap["critic_opt"])
        if self.norm is not None and snap["norm"] is not None:
            self.norm.load_state_dict(snap["norm"])
        self.global_step = int(snap["global_step"])

    def _exploiter_role(self):
        if self.cfg.milestone_period <= 0:
            return "ME-EIE"
        slot = max(0, int(getattr(self, "iteration", 0)) // self.cfg.milestone_period - 1)
        return EXPLOITER_ACTIVE_ROLES[slot % len(EXPLOITER_ACTIVE_ROLES)]

    def _select_exploiter_profile(self, role):
        # The intervention is keyed to the main iteration, so it remains
        # deterministic across a recovery/resume and does not contaminate the
        # role-local adaptive evidence collected by ordinary milestones.
        if int(getattr(self, "iteration", 0)) in self.cfg.forced_altitude_hunt_iterations:
            return "altitude_hunt", ALTITUDE_HUNT_REWARD
        names = PROFILE_NAMES
        if self.archive is None:
            mode = scheduled_exploiter_mode(getattr(self, "iteration", 0), self.cfg.milestone_period,
                                            self.cfg.exploiter_alternate_altitude_hunt)
            return reward_mode_name(mode), mode
        if role not in self._role_profile_bandit:
            raise ValueError(f"unknown exploiter role for profile selection: {role}")
        # Evidence is intentionally isolated by role.  An altitude-hunt LE
        # attempt is not evidence that current-main ME altitude coverage exists.
        stats = self._role_profile_bandit[role]
        total = sum(x["count"] for x in stats.values())
        # Every role must directly try every reward profile once before its
        # adaptive rule can exploit accumulated evidence.  This is a bounded
        # four-attempt coverage guard, not a second scheduler.
        untried = [name for name in names if stats[name]["count"] == 0]
        if untried:
            name = untried[0]
        if role == "ME-EIE":
            # Exploit a demonstrated weakness, with UCB exploration while data is sparse.
            def utility(name):
                item = stats[name]
                return item["utility"] + math.sqrt(math.log(total + 2.0) / (item["count"] + 1.0))
            if not untried:
                name = max(names, key=utility) if total else "standard"
        elif role == "ME-ERE":
            if not untried:
                name = min(names, key=lambda x: (stats[x]["count"], stats[x]["utility"]))
        else:
            if not untried:
                # LE coverage is role-local as well.  Admissions produced by
                # ME-EIE/ME-ERE must not suppress a profile in the LE cell.
                # Prefer the least-tried LE profile and use higher observed
                # utility only as a tie-breaker.
                name = min(names, key=lambda x: (
                    stats[x]["count"], -stats[x]["utility"]))
        mode = {"standard": STANDARD_REWARD, "altitude_hunt": ALTITUDE_HUNT_REWARD,
                "attack": ATTACK_REWARD, "defense": DEFENSE_REWARD}[name]
        return name, mode

    def _select_exploiter_init(self, role, current_bundle):
        if self.archive is None or role == "ME-EIE":
            return current_bundle, "latest_main"
        candidates = [i for i in self.archive.strategic_ids()
                      if not self.archive.records[i].get('metrics', {}).get('external_frozen_only')]
        if not candidates:
            return current_bundle, "latest_main_fallback"
        if role == "LE":
            identity = max(candidates, key=lambda x: self.archive.records[x].get("nash_mass", 0.0))
            return self.archive.load_policy(identity), f"nash_support:{identity}"
        # Explorative ME starts from the payoff fingerprint with the largest
        # variance, rather than returning to one fixed early-main basin.
        def diversity(identity):
            row = self.archive.payoff.get(str(identity), {})
            values = [float(row.get(str(anchor), {}).get("score", 0.5)) for anchor in candidates]
            return float(np.std(values))
        identity = max(candidates, key=lambda x: (diversity(x), -self.archive.records[x]["iteration"]))
        return self.archive.load_policy(identity), f"diverse_anchor:{identity}"

    def _candidate_novelty(self, scores):
        if self.archive is None or not scores:
            return 0.0
        anchors = sorted(scores)
        candidate = np.asarray([scores[x]["score"] for x in anchors], dtype=np.float64)
        distances = []
        for identity in self.archive.strategic_ids():
            row = self.archive.payoff.get(str(identity), {})
            fingerprint = np.asarray([float(row.get(str(x), {}).get("score", 0.5)) for x in anchors])
            distances.append(float(np.linalg.norm(candidate - fingerprint) / math.sqrt(len(anchors))))
        return min(distances) if distances else 1.0

    def _active_family_incumbent(self, role, profile):
        """Strongest active exploiter with the same role x profile label.

        This label is only a bounded successor comparison family.  It is not a
        claim that the two policies are behaviorally identical.
        """
        candidates = []
        for entry in self.pool.active_entries():
            if entry.get("role") not in ("core", "challenger"):
                continue
            identity = entry.get("archive_id")
            if identity is None or identity not in self.archive.records:
                continue
            record = self.archive.records[identity]
            metrics = record.get("metrics") or {}
            if (record.get("admitted")
                    and record.get("kind", "").startswith("exploiter")
                    and metrics.get("role") == role
                    and record.get("profile") == profile):
                # entry EMA is the current main's score.  Lower therefore means
                # a stronger incumbent opponent; prefer newer only on a tie.
                candidates.append((float(entry.get("ema", 0.5)),
                                   -int(record.get("iteration", 0)), int(identity)))
        return min(candidates)[2] if candidates else None

    def _evaluate_and_admit_exploiter(self, candidate, target, role, profile, training_ema):
        seed_block = int(getattr(self, "iteration", 0))
        target_eval = self._evaluate_pair(candidate, target, self.cfg.league_admission_games,
                                          paired=True, seed_block=seed_block)
        anchors = [e.get("archive_id") for e in self.pool.active_entries()
                   if e.get("role") in ("core", "challenger") and e.get("archive_id") is not None]
        anchors = list(dict.fromkeys(anchors))[:4]
        anchor_scores = {}
        for identity in anchors:
            anchor_scores[identity] = self._evaluate_pair(
                candidate, self.archive.load_policy(identity),
                self.cfg.league_payoff_games, paired=True, seed_block=seed_block)
        novelty = self._candidate_novelty(anchor_scores)
        best_anchor = max(anchor_scores.values(), key=lambda x: x["score"], default=None)
        counter_max = float(best_anchor["score"]) if best_anchor is not None else 0.0
        counter_lcb = float(best_anchor["lcb95"]) if best_anchor is not None else 0.0
        anchor_mean = float(np.mean([x["score"] for x in anchor_scores.values()])) if anchor_scores else 0.0
        target_confident = (target_eval["score"] >= self.cfg.league_admission_score
                            and target_eval["lcb95"] >= self.cfg.league_admission_lcb)
        league_contribution = (role == "LE" and counter_max >= 0.70
                               and counter_lcb >= self.cfg.league_admission_lcb
                               and anchor_mean >= 0.50 and novelty >= 0.15)
        accepted = target_confident or league_contribution
        similar_successor = False
        incumbent_id = self._active_family_incumbent(role, profile)
        incumbent_eval = None
        successor_screen = False
        # Normal admission remains unchanged.  Only a rejected candidate gets
        # this one bounded chance against the strongest active member of the
        # same role x profile family.  Both policies use the exact same frozen
        # main, evaluation stage and mirrored scenario seed bank.
        if not accepted and incumbent_id is not None:
            incumbent_eval = self._evaluate_pair(
                self.archive.load_policy(incumbent_id), target,
                self.cfg.league_admission_games, paired=True, seed_block=seed_block)
            successor_screen = bool(target_eval["score"] > incumbent_eval["score"])
            similar_successor = bool(
                successor_screen and target_eval["lcb95"] > incumbent_eval["ucb95"])
            accepted = similar_successor
        if target_confident:
            admission_route = "target_confident"
        elif league_contribution:
            admission_route = "league_contribution"
        elif similar_successor:
            admission_route = "similar_successor"
        else:
            admission_route = "rejected"
        metrics = {"role": role, "training_ema": float(training_ema),
                   "target_score": target_eval["score"], "target_lcb95": target_eval["lcb95"],
                   "target_ucb95": target_eval["ucb95"],
                   "target_games": target_eval["games"], "payoff_novelty": novelty,
                   "successor_family": {"role": role, "profile": profile},
                   "successor_of": incumbent_id,
                   "candidate_alt_loss_rate": target_eval["alt_loss_rate"],
                   "target_main_alt_loss_rate": target_eval["opponent_alt_loss_rate"],
                   "wins_by_target_main_altitude":
                       target_eval["wins_by_opponent_altitude"],
                   "wins_by_target_main_altitude_rate":
                       target_eval["wins_by_opponent_altitude_rate"],
                   "wins_by_target_main_altitude_share":
                       target_eval["wins_by_opponent_altitude_share"],
                   "losses_by_candidate_altitude":
                       target_eval["losses_by_own_altitude"],
                   "losses_by_candidate_altitude_rate":
                       target_eval["losses_by_own_altitude_rate"],
                   "altitude_diagnostic_protocol":
                       target_eval["altitude_diagnostic_protocol"],
                   "anchor_mean": anchor_mean, "anchor_counter_max": counter_max,
                   "anchor_counter_lcb95": counter_lcb,
                   "target_confident": bool(target_confident),
                   "league_contribution_confident": bool(league_contribution),
                   "similar_successor": bool(similar_successor),
                   "successor_screen": bool(successor_screen),
                   "successor_incumbent_score": (
                       None if incumbent_eval is None else float(incumbent_eval["score"])),
                   "successor_incumbent_lcb95": (
                       None if incumbent_eval is None else float(incumbent_eval["lcb95"])),
                   "successor_incumbent_ucb95": (
                       None if incumbent_eval is None else float(incumbent_eval["ucb95"])),
                   "admission_route": admission_route,
                   "admission_rule": "point_lcb_or_similar_successor_v3"}
        archive_id = None
        if accepted:
            kind = (f"exploiter_successor_{role.lower()}"
                    if similar_successor else f"exploiter_{role.lower()}")
            archive_id = self._archive_bundle(
                candidate, kind, profile=profile,
                admitted=True, payoff_eligible=True, metrics=metrics)
            if similar_successor:
                # The incumbent is untouched.  The new policy alone receives a
                # probationary label and, by being the newest admitted
                # exploiter, one of the bounded latest challenger slots.
                self.archive.records[archive_id]["admission_status"] = "probationary"
                self.archive.persist()
            if self._last_milestone_archive_id is not None:
                self._record_payoff_result(archive_id, self._last_milestone_archive_id,
                                           target_eval, seed_block=seed_block)
            for identity, result in anchor_scores.items():
                self._record_payoff_result(archive_id, identity, result,
                                           seed_block=seed_block)
            self._refresh_payoff_row(archive_id, seed_block=seed_block)
        elif target_eval["score"] >= 0.50 or novelty >= 0.20:
            # Useful but non-admitted policies form a held-out red team.  They
            # are never sampled by main unless a later audit proves a regression.
            archive_id = self._archive_bundle(
                candidate, f"heldout_{role.lower()}", profile=profile,
                admitted=False, payoff_eligible=True, metrics=metrics)
        _record_role_profile_result(
            self._role_profile_bandit, role, profile, target_eval["score"], novelty)
        return accepted, archive_id, metrics

    # ── exploiter 학습: ME-EIE / LE / ME-ERE with frozen admission ────────────
    def train_exploiter(self, log=None, metric_cb=None):
        """Train a side learner without discarding the main's ongoing games.

        Main FDM/reconstruction/GRU/assignments/RNG are restored even on failure.
        A new exploiter is eligible only for subsequently starting episodes.
        """
        if self.cfg.exploiter_iters <= 0:
            return None
        self._refresh_weights()
        saved = self._snapshot_learner()
        # Legacy side-learning intentionally appends its exploiter directly to
        # the old pool.  Active league admission goes through the cold archive,
        # so only that protocol restores the paused main's exact resident roster.
        saved_roster = self._pool_roster_snapshot() if self.archive is not None else None
        runtime = self._runtime_state()
        base_ent, base_clip = self.cfg.ent_coef, self.cfg.clip_coef
        base_mode = getattr(self.env, "reward_mode", STANDARD_REWARD)
        base_hunt_coef = getattr(self.env, "alt_hunt_coef", 5.0)
        role = self._exploiter_role()
        profile, mode = self._select_exploiter_profile(role)
        target_bundle = self._policy_bundle()
        init_bundle, init_label = self._select_exploiter_init(role, target_bundle)
        try:
            self.env.reward_mode = mode
            self.env.alt_hunt_coef = self.cfg.exploiter_alt_hunt_coef
            print(f"[gpu-ppo] exploiter role={role} reward={profile} init={init_label} "
                  f"@main{getattr(self, 'iteration', 0)}; target EMA={self.cfg.exploiter_win_target:g}",
                  flush=True)
            return self._train_exploiter_inner(
                role=role, profile=profile, init_bundle=init_bundle,
                target_bundle=target_bundle, log=log, metric_cb=metric_cb)
        finally:
            self.env.reward_mode, self.env.alt_hunt_coef = base_mode, base_hunt_coef
            self.cfg.ent_coef, self.cfg.clip_coef = base_ent, base_clip
            self._restore_learner(saved)
            if saved_roster is not None:
                self._restore_pool_roster(saved_roster)
            self._restore_runtime(runtime, allow_new_pool_entries=True)
            self._refresh_weights()

    def _train_exploiter_inner(self, role="ME-EIE", profile="standard", init_bundle=None,
                               target_bundle=None, log=None, metric_cb=None):
        """metric_cb(i, metrics_dict): 매 exploiter iteration 의 전체 지표를 넘겨(wandb 섹터
        로깅용). log(i, wr, eps): 콘솔 출력용 간단 콜백(기존 호환)."""
        cfg = self.cfg
        if cfg.exploiter_iters <= 0:
            return None
        target_bundle = target_bundle or self._policy_bundle()
        frozen_net, frozen_norm, frozen_state = self._frozen_policy(target_bundle)
        cur_it = int(getattr(self, "iteration", 0))
        init_bundle = init_bundle or target_bundle
        self._load_policy_bundle(init_bundle)
        cfg.ent_coef, cfg.clip_coef = cfg.exploiter_ent_coef, cfg.exploiter_clip_coef
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=cfg.exploiter_lr, eps=1e-5)
        clr = cfg.critic_lr if cfg.critic_lr is not None else cfg.exploiter_lr
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)
        self._reset_env_state()

        # exploiter(=학습 중인 self.model)의 frozen-main 상대 승률 EMA. pool 엔트리와 동일하게
        # 중립값 0.5 에서 시작하고 selfplay_ema_alpha 로 갱신한다. 이 EMA 가 target 을 넘으면
        # "안정적으로 main 을 압도" 로 보고 exploiter 학습을 멈춰 pool 에 추가한다.
        wr_ema = 0.5
        wr = 0.0
        # P0 fix (audit 2026-09-02): LE trains and is scored entirely against
        # the league mixture, which already contains the latest/recent main
        # clones at ~25% weight, plus an LE init drawn from the top Nash-mass
        # policy. wr_ema therefore starts far above 0.5 and used to cross
        # exploiter_win_target within single-digit iterations (observed
        # median 7/1000, 9/9 LE runs rejected on independent admission).
        # Require a minimum amount of *decided* evidence against that mixture
        # before the EMA is trusted to end training early. Reusing
        # league_admission_games keeps this aligned with the games/order the
        # independent admission evaluator already demands, without adding a
        # new checkpoint-contract field.
        le_min_decided_games = float(cfg.league_admission_games) if role == "LE" else 0.0
        le_decided_games_total = 0.0
        # P0 fix (audit 2026-09-02): opponent identity used to change mid
        # in-flight episode whenever the curriculum branch below flipped
        # opp_kind, because live episodes are not reset on every exploiter
        # iteration (episodes average ~1000 steps against a 64-104 step
        # rollout). Reset the environment whenever the *category* of opponent
        # changes so every completed episode was played against one coherent
        # opponent source; the wr_ema/games credited to that source are then
        # unambiguous. LE never switches category, so this adds no reset cost
        # there; ME-EIE/ME-ERE only pay it while a curriculum block is active
        # (see `_curriculum_probability` above for the soft band that decides
        # this, redrawn once per `EXPLOITER_CURRICULUM_BLOCK_ITERS` block).
        previous_opponent_category = None
        curriculum_block_active = False
        # A curriculum block never updates wr_ema (target_iteration=False), so
        # letting curriculum trigger again purely from a *stale* wr_ema could
        # strand the exploiter in curriculum indefinitely -- once it enters,
        # nothing would ever refresh the estimate that keeps it there. Forbid
        # two curriculum blocks back to back: after a curriculum block, the
        # next block always plays frozen-main first (refreshing wr_ema)
        # before the probability gate is consulted again.
        previous_block_was_curriculum = False
        for i in range(1, int(cfg.exploiter_iters) + 1):
            t0 = time.time()
            target_iteration = True
            if self.archive is not None and role == "LE":
                opponent_category = "league_mixture"
                self.opp_weights = self.pool.league_exploiter_weights()
                opponent_altitude_scope = "active_league_mixture"
            else:
                if self.archive is not None and (i - 1) % EXPLOITER_CURRICULUM_BLOCK_ITERS == 0:
                    if previous_block_was_curriculum:
                        curriculum_block_active = False
                    else:
                        p_curriculum = _curriculum_probability(wr_ema)
                        curriculum_block_active = bool(torch.rand(()).item() < p_curriculum)
                    previous_block_was_curriculum = curriculum_block_active
                if self.archive is not None and curriculum_block_active:
                    opponent_category = "variance_curriculum"
                    self.opp_weights = self.pool.variance_curriculum_weights()
                    opponent_altitude_scope = "variance_curriculum_mixture"
                else:
                    opponent_category = "frozen_main"
                    opponent_altitude_scope = "frozen_current_main"
            if opponent_category != previous_opponent_category:
                self._reset_env_state()
            previous_opponent_category = opponent_category
            if opponent_category == "league_mixture":
                adv, ret, rs = self.collect_rollout(opp_kind="pool")
            elif opponent_category == "variance_curriculum":
                # Curriculum only; do not credit these games to target EMA.
                adv, ret, rs = self.collect_rollout(opp_kind="pool")
                target_iteration = False
            else:
                adv, ret, rs = self.collect_rollout(
                    opp_kind="frozen", frozen_opp=(frozen_net, frozen_norm, frozen_state))
            u = self.update(adv, ret)
            require_finite(u, "exploiter update")
            ep_c = float(rs["ep_count"])
            dec = float(rs["win_sum"] + rs["loss_sum"])
            wr = float(rs["win_sum"]) / dec if dec > 0 else 0.0
            # 무승부 0.5 반영 win_rate(리포팅용; main 과 동일 규약).
            n_draw = max(0.0, ep_c - float(rs["win_sum"]) - float(rs["loss_sum"]))
            wr_draw = (float(rs["win_sum"]) + 0.5 * n_draw) / ep_c if ep_c > 0 else float("nan")
            mean_ret = float(rs["ret_sum"] / rs["ep_count"]) if ep_c > 0 else float("nan")
            alt_lr = float(rs["alt_loss_sum"]) / ep_c if ep_c > 0 else float("nan")
            target_alt_lr = float(rs["opponent_alt_loss_sum"]) / ep_c if ep_c > 0 else float("nan")
            # 승률 EMA 갱신(완료 에피소드 있을 때만; 무승부 0.5 반영 규약 동일). ep_c==0 이면
            # 직전 EMA 유지(정보 없음).
            if ep_c > 0 and target_iteration:
                # Active-league protocol uses a game-count half-life so that an
                # iteration containing 5 completed games cannot move the estimate
                # as much as one containing 500.  Preserve the original fixed-alpha
                # rule for explicitly legacy (league-disabled) runs/checkpoints.
                alpha = (count_aware_alpha(ep_c, cfg.league_ema_half_life_games)
                         if self.archive is not None else float(cfg.selfplay_ema_alpha))
                wr_ema = (1.0 - alpha) * wr_ema + alpha * wr_draw
                if role == "LE":
                    le_decided_games_total += dec
            dt = time.time() - t0
            # 매 iteration 콘솔 출력(main 루프와 비슷한 정보량).
            print(f"[gpu-ppo]   exp it {i:4d} | wr_ema {wr_ema:.3f} wr {wr:.3f}(draw {wr_draw:.3f}) "
                  f"| eps {int(ep_c):4d} ret {mean_ret:7.2f} altL {alt_lr:.3f} "
                  f"oppAltL {target_alt_lr:.3f}({opponent_altitude_scope}) "
                  f"| pl {float(u['pl']):+.3f} vl {float(u['vl']):.3f} ent {float(u['ent']):.3f} "
                  f"kl {float(u['kl']):.4f} | {dt:.1f}s", flush=True)
            if log is not None:
                log(i, wr, ep_c)
            if metric_cb is not None:
                metric_cb(i, {
                    "reward_mode": int(self.env.reward_mode),
                    "role": role, "profile": profile,
                    "target_iteration": bool(target_iteration),
                    # New key intentionally avoids merging the corrected pool
                    # measurement with historical LE values that were logged
                    # as a synthetic zero under ``opponent_alt_loss_rate``.
                    "observed_opponent_alt_loss_rate_v2": target_alt_lr,
                    "opponent_alt_loss_scope_v2": opponent_altitude_scope,
                    "opponent_alt_loss_valid_v2": True,
                    "win_rate": wr_draw, "win_rate_decided": wr, "win_rate_ema": wr_ema,
                    "completed_episodes": ep_c,
                    "mean_return": mean_ret, "alt_loss_rate": alt_lr,
                    "policy_loss": float(u["pl"]), "value_loss": float(u["vl"]),
                    "entropy": float(u["ent"]), "approx_kl": float(u["kl"]),
                    "clipfrac": float(u["cf"]), "explained_variance": float(u["ev"]),
                    "ent_coef": float(cfg.ent_coef), "clip_coef": float(cfg.clip_coef),
                    "lr": float(cfg.exploiter_lr), "elapsed_sec": dt,
                    **{key: float(value) for key, value in u.items() if key.startswith("aux_")},
                })
            # 게이팅: 판정 승률이 아니라 승률 EMA 가 target 을 넘으면 조기 종료.
            # LE 는 리그 혼합(=자기 자신의 latest/recent 사본을 포함) 상대로 EMA 를 재므로
            # 최소 판수(league_admission_games) 이상 결판이 나기 전에는 조기 종료하지 않는다.
            enough_le_evidence = (role != "LE"
                                  or le_decided_games_total >= le_min_decided_games)
            if wr_ema >= cfg.exploiter_win_target and enough_le_evidence:
                print(f"[gpu-ppo]   exploiter 조기 종료: wr_ema {wr_ema:.3f} ≥ "
                      f"target {cfg.exploiter_win_target} (it {i}"
                      + (f", LE decided games {le_decided_games_total:.0f}/"
                         f"{le_min_decided_games:.0f}" if role == "LE" else "") + ")",
                      flush=True)
                break

        candidate = self._policy_bundle()
        if self.archive is not None:
            accepted, archive_id, admission = self._evaluate_and_admit_exploiter(
                candidate, target_bundle, role, profile, wr_ema)
            print(f"[gpu-ppo]   exploiter admission accepted={accepted} archive_id={archive_id} "
                  f"score={admission['target_score']:.3f} lcb95={admission['target_lcb95']:.3f} "
                  f"novelty={admission['payoff_novelty']:.3f} "
                  f"targetMainAltL={admission['target_main_alt_loss_rate']:.3f} "
                  f"altitudeWins={admission['wins_by_target_main_altitude']}/"
                  f"{admission['target_games']}", flush=True)
            if metric_cb is not None:
                metric_cb(i + 1, {
                    "event": "independent_admission_evaluation",
                    "role": role, "profile": profile,
                    "admission_accepted": bool(accepted),
                    "admission_target_score": admission["target_score"],
                    "admission_target_lcb95": admission["target_lcb95"],
                    "admission_target_games": admission["target_games"],
                    "admission_candidate_alt_loss_rate_v2":
                        admission["candidate_alt_loss_rate"],
                    "admission_target_main_alt_loss_rate_v2":
                        admission["target_main_alt_loss_rate"],
                    "admission_wins_by_target_main_altitude_v2":
                        admission["wins_by_target_main_altitude"],
                    "admission_wins_by_target_main_altitude_rate_v2":
                        admission["wins_by_target_main_altitude_rate"],
                    "admission_losses_by_candidate_altitude_v2":
                        admission["losses_by_candidate_altitude"],
                })
            history = {"main_iteration": cur_it, "role": role, "reward_mode": profile,
                       "iterations": i, "win_rate_ema": wr_ema,
                       "win_target": cfg.exploiter_win_target, "accepted": accepted,
                       "archive_id": archive_id, "admission": admission,
                       "diagnostic_protocol":
                           "opponent_altitude_outcome_observability_v2"}
        else:
            self.pool.add(self.model, self.norm, permanent=True, ema=0.5)
            history = {"main_iteration": cur_it, "reward_mode": reward_mode_name(self.env.reward_mode),
                       "iterations": i, "win_rate_ema": wr_ema,
                       "win_target": cfg.exploiter_win_target,
                       "pool_opponent_id": self.pool.next_id - 1}
        self.exploiter_history.append(history)

        return wr_ema

    # ── 메인 루프 ────────────────────────────────────────────────────────────
    def train(self, on_iteration: Optional[Callable[[GPUIterationStats], None]] = None,
              start_iteration=1, on_exploiter_iter: Optional[Callable[[int, int, dict], None]] = None):
        """on_exploiter_iter(milestone_it, exp_iter, metrics): milestone 에서 도는 exploiter
        학습의 매 iteration 지표를 넘긴다(wandb 섹터별 로깅용)."""
        history = []
        it = start_iteration
        while self.cfg.total_iterations <= 0 or it <= self.cfg.total_iterations:
            self.iteration = it
            self._apply_schedule(it)   # 2000-iter 스케줄: lr·ent_coef 감쇠, rollout 증가
            t0 = time.time()
            adv, ret, rstats = self.collect_rollout(opp_kind="pool")
            u = self.update(adv, ret)

            # 이 update 직후 main net 은 정확히 'it 번 학습된' 버전. exploiter 초기화 시드는
            # first(기본 500)/rest(기본 1000) iter 시점에 각각 1회만 캡처해 이후 재사용한다.
            if self.archive is None and self._exp_init_first is None and \
                    it == int(getattr(self.cfg, "exploiter_init_iteration_first", 0)):
                self._exp_init_first = self._capture_exploiter_init()
                print(f"[gpu-ppo] exploiter(first) 초기화 스냅샷 캡처: iter{it} main net", flush=True)
            if self.archive is None and self._exp_init_rest is None and \
                    it == int(getattr(self.cfg, "exploiter_init_iteration_rest", 0)):
                self._exp_init_rest = self._capture_exploiter_init()
                print(f"[gpu-ppo] exploiter(rest) 초기화 스냅샷 캡처: iter{it} main net", flush=True)

            ep_count = float(rstats["ep_count"])
            mean_ret = float(rstats["ret_sum"] / rstats["ep_count"]) if ep_count > 0 else float("nan")
            mean_len = float(rstats["len_sum"] / rstats["ep_count"]) if ep_count > 0 else float("nan")
            # 리포팅 win_rate: 무승부(draw=완료 - 승 - 패)를 0.5 로 계산 → 완료 에피소드가
            # 있으면 nan 이 나오지 않는다. 분모는 전체 완료 에피소드(승+패+무).
            n_win = float(rstats["win_sum"]); n_loss = float(rstats["loss_sum"])
            n_draw = max(0.0, ep_count - n_win - n_loss)
            win_rate = (n_win + 0.5 * n_draw) / ep_count if ep_count > 0 else float("nan")
            # main 이 고도제한 위반으로 패한 게임 비율(완료 에피소드 대비). exploiter 학습은
            # 별도 루프(train_exploiter)라 on_iteration 을 안 타므로 자동으로 제외된다.
            alt_loss_rate = float(rstats["alt_loss_sum"]) / ep_count if ep_count > 0 else float("nan")
            elapsed = time.time() - t0
            sps = self.cfg.rollout_steps * self.nenv / max(elapsed, 1e-9)
            pool_event = None
            payoff_refresh_due = bool(
                self.archive is not None
                and self.cfg.league_payoff_refresh_period > 0
                and it % self.cfg.league_payoff_refresh_period == 0)
            payoff_refreshed = False

            # EMA 갱신 → 게이팅(evictable 최소 EMA ≥ threshold 면 현재 main 추가).
            self.pool.update_emas(rstats["win_by_opp"], rstats["loss_by_opp"],
                                  rstats["ep_by_opp"], self.cfg.selfplay_ema_alpha,
                                  opponent_ids=rstats["opponent_ids"])
            if self.archive is not None:
                pool_event = self._maybe_update_latest_and_recent(it)
            elif self.pool.gate_and_add(self.model, self.norm, self.cfg.selfplay_gate_threshold):
                pool_event = "gate"
            ema_min = min((e["ema"] for e in self.pool.active_entries() if not e["permanent"]), default=float("nan"))
            ema_mean = float(np.mean([e["ema"] for e in self.pool.active_entries()])) if self.pool.size() else float("nan")

            # milestone: permanent main snapshot(+capacity) + exploiter.
            if self.cfg.milestone_period > 0 and it % self.cfg.milestone_period == 0:
                if self.archive is not None:
                    # P2 fix (audit 2026-09-02): when recent_period divides
                    # milestone_period (both default 500/100), the recent-ring
                    # update above already archived a byte-identical snapshot
                    # of this exact main at this exact iteration a few lines
                    # earlier. Reuse that archive record instead of writing a
                    # second, bit-identical policy file: flip it eligible
                    # rather than re-archiving. It is excluded from
                    # select_core()'s candidates while still resident as
                    # "recent" (see ``_refresh_active_core``'s ``exclude``),
                    # so this cannot make it enter core twice.
                    reused_recent_id = None
                    if pool_event == "recent" and self._recent_archive_ids:
                        candidate_id = self._recent_archive_ids[-1]
                        candidate_record = self.archive.records.get(candidate_id)
                        if (candidate_record is not None
                                and int(candidate_record.get("iteration", -1)) == int(it)):
                            reused_recent_id = candidate_id
                    if reused_recent_id is not None:
                        record = self.archive.records[reused_recent_id]
                        record["kind"] = "milestone_main"
                        record["payoff_eligible"] = True
                        self._last_milestone_archive_id = reused_recent_id
                    else:
                        self._last_milestone_archive_id = self._archive_current(
                            "milestone_main", admitted=True, payoff_eligible=True)
                    payoff_block = it // max(1, self.cfg.league_payoff_refresh_period)
                    self._refresh_payoff_row(self._last_milestone_archive_id,
                                             seed_block=payoff_block)
                    if payoff_refresh_due:
                        self._refresh_uncertain_payoff_edge(seed_block=payoff_block)
                        payoff_refreshed = True
                    # The side learner must see the league assembled from the
                    # newly updated payoff matrix.  In particular, the first
                    # LE milestone excludes the *current* milestone from core
                    # selection and should therefore train against the prior
                    # milestone(s), not fall back to latest/recent only.
                    self._refresh_active_core()
                    pool_event = "archive_milestone"
                else:
                    self.pool.add(self.model, self.norm, permanent=True, ema=0.5)
                    pool_event = "milestone"
                side_learner_due = (
                    self.cfg.exploiter_iters > 0
                    and (self.cfg.exploiter_until_iteration <= 0
                         or it <= self.cfg.exploiter_until_iteration)
                )
                if side_learner_due:
                    print(f"[gpu-ppo] === exploiter 학습 시작 @it{it} "
                          f"(max {self.cfg.exploiter_iters}it, target wr_ema ≥ {self.cfg.exploiter_win_target}) ===",
                          flush=True)
                    _mcb = ((lambda i, m: on_exploiter_iter(it, i, m))
                            if on_exploiter_iter is not None else None)
                    ewr = self.train_exploiter(metric_cb=_mcb)  # 매 iter 출력은 내부에서 처리
                    print(f"[gpu-ppo] === exploiter 완료 wr_ema {ewr:.3f}, pool {self.pool.size()} "
                          f"(perm {self.pool.num_permanent()}, cap {self.pool.capacity()}) ===", flush=True)
                elif self.cfg.exploiter_iters > 0:
                    print(f"[gpu-ppo] === exploiter skipped @it{it}: "
                          f"configured through {self.cfg.exploiter_until_iteration}; "
                          "archive/core maintenance continues ===", flush=True)
                if self.archive is not None:
                    # Re-select after side-learner admission so newly accepted
                    # challengers are available to subsequent main episodes.
                    self._refresh_active_core()

            # These schedules are independent of milestone cadence.  When they
            # coincide with a milestone, the new row is committed first and the
            # side learner sees the refreshed core before training.
            if self.archive is not None and payoff_refresh_due and not payoff_refreshed:
                payoff_block = it // self.cfg.league_payoff_refresh_period
                self._refresh_uncertain_payoff_edge(seed_block=payoff_block)
                self._refresh_active_core()
            if self.archive is not None:
                promoted = self._audit_heldout_redteam(it)
                if promoted:
                    pool_event = "redteam_promoted:" + ",".join(map(str, promoted))
                    self._refresh_active_core()

            self._refresh_weights()

            # 슬롯별 EMA(리포팅용): evict 슬롯은 FIFO 위치 기준, perm 은 추가 순.
            evict_slot_emas, perm_slot_emas = self.pool.slot_emas()

            role_counts = self.pool.role_counts() if isinstance(self.pool, ActiveLeaguePool) else {}
            stats = GPUIterationStats(
                iteration=it, global_step=self.global_step,
                mean_return=mean_ret, mean_length=mean_len, completed_episodes=ep_count,
                win_rate=win_rate,
                policy_loss=float(u["pl"]), value_loss=float(u["vl"]),
                entropy=float(u["ent"]), approx_kl=float(u["kl"]), clipfrac=float(u["cf"]),
                explained_variance=float(u["ev"]), steps_per_sec=sps, elapsed_sec=elapsed,
                extra={"epochs": int(u["epochs"]), "early_stop": bool(u["early"]),
                       "actor_grad_norm": float(u["actor_grad_norm"]),
                       "critic_grad_norm": float(u["critic_grad_norm"]),
                       "optimizer_steps": int(u["optimizer_steps"]),
                       "pool_size": self.pool.size(), "pool_perm": self.pool.num_permanent(),
                       "pool_cap": self.pool.capacity(), "pool_event": pool_event,
                       "pool_resident": self.pool.resident_size(),
                       "pool_retired": self.pool.resident_size() - self.pool.size(),
                       "archive_size": len(self.archive) if self.archive is not None else 0,
                       "league_latest": role_counts.get("latest", 0),
                       "league_recent": role_counts.get("recent", 0),
                       "league_core": role_counts.get("core", 0),
                       "league_challenger": role_counts.get("challenger", 0),
                       "ema_min": float(ema_min), "ema_mean": ema_mean,
                       "evict_slot_emas": evict_slot_emas, "perm_slot_emas": perm_slot_emas,
                       "alt_loss_rate": alt_loss_rate,
                       "lr": float(self.actor_opt.param_groups[0]["lr"]),
                       "ent_coef": float(self.cfg.ent_coef),
                       "rollout": int(self.cfg.rollout_steps)})
            stats.extra.update({key: float(value) for key, value in u.items() if key.startswith("aux_")})
            if self.cfg.aux_pred:
                require_finite({key: value for key, value in stats.extra.items()
                                if key.startswith("aux_")}, "auxiliary iteration metrics")
            history.append(stats)
            require_finite_training_stats(stats)
            if on_iteration is not None:
                on_iteration(stats)
            it += 1
        return history

    # ── checkpoint ───────────────────────────────────────────────────────────
    def save(self, path):
        ckpt = {"training_protocol": self.training_protocol,
                "reward_contract": REWARD_CONTRACT,
                "main_schedule": self._schedule_contract(),
                "league_contract": self._league_contract(),
                "league_archive": self.archive.state_dict() if self.archive is not None else None,
                "league_runtime": ({"latest_clone_iteration": self._latest_clone_iteration,
                                    "recent_archive_ids": list(self._recent_archive_ids),
                                    "last_milestone_archive_id": self._last_milestone_archive_id,
                                    "heldout_audit_cursor": self._heldout_audit_cursor,
                                    "profile_scheduler_protocol": PROFILE_SCHEDULER_PROTOCOL,
                                    "role_profile_bandit": copy.deepcopy(
                                        self._role_profile_bandit)}
                                   if self.archive is not None else None),
                "exploiter_history": self.exploiter_history,
                "model": self.model.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "norm": self.norm.state_dict() if self.norm is not None else None,
                "pool": self.pool.state_dicts(),
                "pool_evict_cap": self.pool.evict_cap,
                "pool_next_id": self.pool.next_id,
                "global_step": self.global_step,
                "iteration": int(getattr(self, "iteration", 0)),
                # exploiter 초기화 시드 2종(resume 보존): first=500-iter, rest=1000-iter.
                "exploiter_init_first": getattr(self, "_exp_init_first", None),
                "exploiter_init_rest": getattr(self, "_exp_init_rest", None),
                "cfg": self.cfg.__dict__}
        if self.cfg.aux_pred:
            ckpt["auxiliary_contract"] = AUX_CONTRACT
        if self.cfg.save_runtime:
            ckpt["runtime"] = self._runtime_state()
        # Do not replace a healthy recovery checkpoint with invalid parameters,
        # optimizer state, reconstruction history or hidden state.
        require_finite({k: ckpt[k] for k in ("model", "actor_opt", "critic_opt", "norm", "pool")},
                       "checkpoint learner")
        if "runtime" in ckpt:
            require_finite(ckpt["runtime"], "checkpoint runtime")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, temp)
        os.replace(temp, path)

    def _runtime_state(self):
        names = ("_next_obs", "_next_opp_obs", "_next_done", "_actor_h", "_critic_h",
                 "opp_assign", "opp_weights", "ep_ret", "ep_len")
        return {"trainer": {n: getattr(self, n).clone() for n in names},
                "sim": self.env.sim.states.clone(), "obr": self.env.obr.clone_state(),
                "sim_buffers": {n: getattr(self.env.sim, n).clone() for n in ("obs", "actions")},
                "ic_pool": self.env.ic_pool,
                "reward_mode": getattr(self.env, "reward_mode", STANDARD_REWARD),
                "alt_hunt_coef": getattr(self.env, "alt_hunt_coef", 5.0),
                "pool_h": {e["id"]: (None if e.get("actor_state") is None else
                                     tuple(h.detach().clone() for h in e["actor_state"]))
                           for e in self.pool.entries},
                # Opponent policies use independent generators so evaluating or
                # side-training cannot perturb the learner RNG.  Their states are
                # part of the main runtime contract and must also be restored after
                # an LE/ME curriculum rollout.
                "pool_rng": {e["id"]: e["rng"].get_state().cpu().clone()
                             for e in self.pool.entries if e.get("rng") is not None},
                "env_rng": copy.deepcopy(self.env.rng.bit_generator.state),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state(self.cfg.device)
                             if torch.device(self.cfg.device).type == "cuda" else None)}

    def _restore_runtime(self, state, allow_new_pool_entries=False):
        require_finite(state, "restored runtime")
        self.env.reward_mode = validate_reward_mode(state["reward_mode"], state["alt_hunt_coef"])
        self.env.alt_hunt_coef = float(state["alt_hunt_coef"])
        resident_ids = {e["id"] for e in self.pool.entries}
        hidden_ids = set(state["pool_h"])
        if hidden_ids != resident_ids and not (allow_new_pool_entries and hidden_ids <= resident_ids):
            raise ValueError("runtime opponent hidden-state IDs do not match the pool")
        rng_ids = set(state.get("pool_rng", {}))
        live_rng_ids = {e["id"] for e in self.pool.entries if e.get("rng") is not None}
        if rng_ids != live_rng_ids and not (allow_new_pool_entries and rng_ids <= live_rng_ids):
            raise ValueError("runtime opponent RNG IDs do not match the pool")
        saved_trainer = state["trainer"]
        if saved_trainer["opp_assign"].shape != (self.nenv,):
            raise ValueError("runtime environment count mismatch")
        live = ~saved_trainer["_next_done"].bool()
        live_ids = set(saved_trainer["opp_assign"][live].detach().cpu().tolist())
        if not live_ids <= resident_ids:
            raise ValueError("runtime live episode refers to a missing opponent ID")
        for name, value in state["trainer"].items():
            setattr(self, name, value.to(self.cfg.device).clone())
        self.env.sim.states.copy_(state["sim"])
        for name, value in state["sim_buffers"].items():
            getattr(self.env.sim, name).copy_(value)
        if 'external_accel' in state['obr']:
            self.env.obr.enable_external_observations()
        self.env.obr.restore(state["obr"])
        self.env.ic_pool = state["ic_pool"]
        for entry in self.pool.entries:
            hidden = state["pool_h"].get(entry["id"])
            entry["actor_state"] = (None if hidden is None else
                                     tuple(h.to(self.cfg.device).clone() for h in hidden))
            if entry.get("rng") is not None and entry["id"] in state.get("pool_rng", {}):
                entry["rng"].set_state(state["pool_rng"][entry["id"]].cpu())
        self.env.rng.bit_generator.state = state["env_rng"]
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.cfg.device)

    def load(self, path, map_location=None):
        # 자체 checkpoint(optimizer/cfg/pool 포함)라 weights_only=False (신뢰 소스).
        ckpt = torch.load(path, map_location=map_location or self.cfg.device, weights_only=False)
        if ckpt.get("training_protocol") != self.training_protocol:
            raise ValueError("legacy/different training protocol: weights remain exportable, but "
                             "resuming would mix opponent-lifecycle, finite-horizon targets or minibatch rules; "
                             "use a separately approved experiment, not an implicit migration")
        saved_cfg = ckpt.get("cfg", {})
        if ckpt.get("main_schedule") != self._schedule_contract():
            raise ValueError("checkpoint main schedule differs or is missing; "
                             "resume requires the same base LR/entropy, decay, floors, rollout cap and curriculum")
        if ckpt.get("league_contract") != self._league_contract():
            raise ValueError("checkpoint active-league contract differs or is missing")
        if ckpt.get("reward_contract") != REWARD_CONTRACT:
            raise ValueError("checkpoint reward contract differs; implicit migration refused")
        for key in ("exploiter_alternate_altitude_hunt", "exploiter_alt_hunt_coef", "exploiter_win_target"):
            if saved_cfg.get(key) != getattr(self.cfg, key):
                raise ValueError(f"checkpoint exploiter contract differs: {key}")
        self.exploiter_history = copy.deepcopy(ckpt.get("exploiter_history", []))
        if bool(saved_cfg.get("aux_pred", False)) != bool(self.cfg.aux_pred):
            raise ValueError("checkpoint auxiliary ON/OFF differs from requested training")
        if self.cfg.aux_pred and (ckpt.get("auxiliary_contract") != AUX_CONTRACT
                                  or saved_cfg.get("aux_coef") != self.cfg.aux_coef):
            raise ValueError("checkpoint auxiliary target/loss contract differs; implicit migration refused")
        require_finite({k: ckpt.get(k) for k in ("model", "actor_opt", "critic_opt", "norm", "pool")},
                       "loaded checkpoint")
        if int(ckpt["pool_evict_cap"]) != self.pool.evict_cap:
            raise ValueError("checkpoint pool capacity differs from the requested configuration")
        self.model.load_state_dict(ckpt["model"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        # torch.load(map_location='cuda') also moves Adam's scalar counters.
        # Non-capturable/non-fused Adam deliberately keeps these on CPU; the
        # optimizer loader preserves their loaded device rather than fixing it.
        # Restore original placement (not values) to avoid per-step GPU reads.
        for optimizer in (self.actor_opt, self.critic_opt):
            for group in optimizer.param_groups:
                if group.get("capturable", False) or group.get("fused", False):
                    continue
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    if torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].cpu()
        if self.norm is not None and ckpt.get("norm") is not None:
            self.norm.load_state_dict(ckpt["norm"])
        if self.archive is not None:
            if ckpt.get("league_archive") is None:
                raise ValueError("active-league checkpoint is missing its archive metadata")
            self.archive.load_state_dict(ckpt["league_archive"])
        if ckpt.get("pool") is not None:
            self.pool.load_state_dicts(ckpt["pool"], next_id=ckpt["pool_next_id"])
        if self.archive is not None:
            league_runtime = ckpt.get("league_runtime") or {}
            self._latest_clone_iteration = int(league_runtime["latest_clone_iteration"])
            self._recent_archive_ids = list(league_runtime.get("recent_archive_ids", []))
            self._last_milestone_archive_id = league_runtime.get("last_milestone_archive_id")
            self._heldout_audit_cursor = int(league_runtime.get("heldout_audit_cursor", 0))
            if league_runtime.get("profile_scheduler_protocol") != PROFILE_SCHEDULER_PROTOCOL:
                raise ValueError("checkpoint role/profile scheduler protocol differs or is missing")
            self._role_profile_bandit = _validate_role_profile_bandit(
                copy.deepcopy(league_runtime["role_profile_bandit"]))
        # exploiter 초기화 시드 2종 복원(없으면 None → 해당 iter 에 재캡처/폴백).
        self._exp_init_first = ckpt.get("exploiter_init_first", None)
        self._exp_init_rest = ckpt.get("exploiter_init_rest", None)
        if self._exp_init_rest is None:   # 구 checkpoint 호환: 단일 exploiter_init 키를 rest 시드로.
            self._exp_init_rest = ckpt.get("exploiter_init", None)
        self.global_step = int(ckpt.get("global_step", 0))
        self.iteration = int(ckpt.get("iteration", 0))
        self.opp_weights = self._pool_weights()
        self._reset_env_state()
        if self.cfg.save_runtime and ckpt.get("runtime") is not None:
            self._restore_runtime(ckpt["runtime"])
        self._refresh_weights()
        return ckpt


__all__ = ["PPOGPUConfig", "PPOGPUTrainer", "ActorCritic", "MLPActorCritic",
           "build_actor_critic", "RunningNorm",
           "OpponentPool", "GPUIterationStats", "action_to_env", "make_action_grid",
           "ACTION_BINS"]
