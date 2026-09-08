# -*- coding: utf-8 -*-
"""GPU 벡터화 1v1 dogfight RL env (CUDA FDM 기반).

GpuDogfight(검증된 JSBSim v1.0.0 물리, fp64) 위에 대회 학습 규약을 얹는다:
  - 상태를 공식 9-DOF 규약으로 내보낸다: N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s].
    수평 N/E는 공식 BaseLat/BaseLon 기준 tangent 좌표이며, 수직축은 공식 Unreal
    계약대로 JSBSim MSL(ASL)을 별도 tangent-plane 투영하지 않고 D=-MSL 로 둔다.
    euler 는 FDM 의 로컬-NED euler 그대로, uvw 는 fps×0.3048 이다.
  - 대회 초기배치 분포(env_utils.STANDARD_ENV_CONFIG): 시나리오 A(수직·반대 3) / B(head-on 1),
    고도 2000~30000ft, 속도 200~300m/s, 거리 A={2000,2500,3000}ft·B=10000ft. 두 기체 공유.
  - **staggered 랜덤 리셋**: 첫 iteration(및 학습 재개) 시 각 env 를 랜덤한 스텝수만큼 미리
    진행시켜 에피소드-위상 분포를 고르게 한다(수많은 env 가 t≈0 에 몰리는 편향 제거).

obs(claude164r)/reward 의 GPU 벡터화는 다음 단계. 지금은 9-DOF 상태(state9)를 노출한다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cuda_fdm.gpu_env import GpuDogfight, OBS, STATE_N, OBS_N
from cuda_fdm.obs_reward import BatchObsReward, OBS_SIZE
from claude_code.my_reward import MY_REWARD_CONFIG
from cuda_fdm.finite_checks import require_finite
from cuda_fdm.reward_modes import validate_reward_mode
from dogfight.envs.termination import time_limit_reached

# ── 공식 BaseLat/BaseLon 원점 & 상수 ──────────────────────────────────────────
ORIGIN_LAT_DEG = 37.240778
ORIGIN_LON_DEG = 131.869556
ORIGIN_ALT_M = 0.0
WGS84_A_M = 6378137.0
WGS84_E2 = 0.0066943799901411   # pymap3d wgs84
FT2M = 0.3048
M2FT = 1.0 / FT2M
HARD_DECK_FT = 1000.0
HARD_DECK_M = HARD_DECK_FT * FT2M
D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# state9 컬럼
S9 = dict(N=0, E=1, D=2, roll=3, pitch=4, yaw=5, u=6, v=7, w=8)

# FdmState 컬럼(ic._flatten_state 순서): eci_pos0:3, eci_vel3:6, q6:10, vPQRi10:13, epa13, fuel14
ST_ECI_POS = slice(0, 3)
ST_ECI_VEL = slice(3, 6)
ST_Q = slice(6, 10)
ST_EPA = 13
SEA_LEVEL_RADIUS_FT = 20925646.32546   # = SEMI_MAJOR (alt_asl = radius - a)
ROTATION_RATE = 7.292115e-5


def _geodetic2ecef(lat, lon, h_m):
    """WGS84 geodetic(rad,rad,m) → ecef(m). 배치 텐서. pymap3d 와 동일 공식."""
    slat, clat = torch.sin(lat), torch.cos(lat)
    slon, clon = torch.sin(lon), torch.cos(lon)
    N = WGS84_A_M / torch.sqrt(1.0 - WGS84_E2 * slat * slat)
    x = (N + h_m) * clat * clon
    y = (N + h_m) * clat * slon
    z = (N * (1.0 - WGS84_E2) + h_m) * slat
    return x, y, z


# 원점 ecef (상수)
_o_lat = torch.tensor(ORIGIN_LAT_DEG * D2R, dtype=torch.float64)
_o_lon = torch.tensor(ORIGIN_LON_DEG * D2R, dtype=torch.float64)
_OX, _OY, _OZ = _geodetic2ecef(_o_lat, _o_lon, torch.tensor(ORIGIN_ALT_M, dtype=torch.float64))
_O_SLAT, _O_CLAT = math.sin(ORIGIN_LAT_DEG * D2R), math.cos(ORIGIN_LAT_DEG * D2R)
_O_SLON, _O_CLON = math.sin(ORIGIN_LON_DEG * D2R), math.cos(ORIGIN_LON_DEG * D2R)


def ecef_to_geodetic_latlon(x, y, z):
    """ecef(m) → geodetic lat,lon(rad). Bowring 폐형(WGS84). 배치."""
    b = WGS84_A_M * math.sqrt(1.0 - WGS84_E2)
    ep2 = (WGS84_A_M * WGS84_A_M - b * b) / (b * b)
    p = torch.sqrt(x * x + y * y)
    lon = torch.atan2(y, x)
    th = torch.atan2(z * WGS84_A_M, p * b)
    s3, c3 = torch.sin(th) ** 3, torch.cos(th) ** 3
    lat = torch.atan2(z + ep2 * b * s3, p - WGS84_E2 * WGS84_A_M * c3)
    return lat, lon


def ned_from_ecef_altasl(ecef_m, alt_asl_m):
    """JSBSim ECEF + radial MSL altitude -> official N/E/D state.

    N/E retain the official-origin tangent projection used by the simulator
    bridge.  D is deliberately *not* the tangent projection's vertical
    component: the contest contract maps Unreal z directly to JSBSim MSL, so
    D=-alt_asl_m.  This prevents Earth-curvature error from entering altitude,
    termination, relative vertical geometry, and reward observations.
    """
    ex, ey, ez = ecef_m[:, 0], ecef_m[:, 1], ecef_m[:, 2]
    lat, lon = ecef_to_geodetic_latlon(ex, ey, ez)
    hx, hy, hz = _geodetic2ecef(lat, lon, alt_asl_m)
    dev = hx.device
    dx, dy, dz = hx - _OX.to(dev), hy - _OY.to(dev), hz - _OZ.to(dev)
    n = -_O_SLAT * _O_CLON * dx - _O_SLAT * _O_SLON * dy + _O_CLAT * dz
    e = -_O_SLON * dx + _O_CLON * dy
    d = -alt_asl_m
    return n, e, d


def _quat_to_T(q):
    """q(nac,4) → Ti2b (nac,3,3). jsb_frames.quat_to_T 벡터화."""
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    q0q1, q0q2, q0q3 = q0 * q1, q0 * q2, q0 * q3
    q1q2, q1q3, q2q3 = q1 * q2, q1 * q3, q2 * q3
    row0 = torch.stack([q0q0 + q1q1 - q2q2 - q3q3, 2 * (q1q2 + q0q3), 2 * (q1q3 - q0q2)], 1)
    row1 = torch.stack([2 * (q1q2 - q0q3), q0q0 - q1q1 + q2q2 - q3q3, 2 * (q2q3 + q0q1)], 1)
    row2 = torch.stack([2 * (q1q3 + q0q2), 2 * (q2q3 - q0q1), q0q0 - q1q1 - q2q2 + q3q3], 1)
    return torch.stack([row0, row1, row2], 1)   # (nac,3,3)


def kinematics_state(st):
    """FdmState (nac,101) → (ecef_m(nac,3), euler_rad(nac,3), vUVW_fps(nac,3), alt_asl_ft(nac)).
    fdm.cuh kinematics 벡터화(정확 동일 규약: geocentric 로컬프레임 euler)."""
    eci = st[:, ST_ECI_POS]; vel = st[:, ST_ECI_VEL]; q = st[:, ST_Q]; epa = st[:, ST_EPA]
    px, py, pz = eci[:, 0], eci[:, 1], eci[:, 2]
    ce, se = torch.cos(epa), torch.sin(epa)
    # ecef = Ti2ec(epa)·eci  (ft)
    ex = ce * px + se * py; ey = -se * px + ce * py; ez = pz
    radius = torch.sqrt(ex * ex + ey * ey + ez * ez)
    rxy = torch.sqrt(ex * ex + ey * ey)
    sinLat = ez / radius; cosLat = rxy / radius
    sinLon = ey / rxy; cosLon = ex / rxy
    # Tl2ec = Tec2l^T,  Tec2l rows (geocentric NED). Tl2i = Tec2i·Tl2ec, Tec2i=Ti2ec^T
    # Tl2i columns = Tec2i·(columns of Tl2ec). 여기선 Tl2b=Ti2b·Tl2i 후 euler 만 필요.
    # Tec2l (3x3):
    Tec2l = torch.stack([
        torch.stack([-cosLon * sinLat, -sinLon * sinLat, cosLat], 1),
        torch.stack([-sinLon, cosLon, torch.zeros_like(ex)], 1),
        torch.stack([-cosLon * cosLat, -sinLon * cosLat, -sinLat], 1)], 1)  # (nac,3,3)
    Tl2ec = Tec2l.transpose(1, 2)
    # Tec2i = Ti2ec^T = [[ce,-se,0],[se,ce,0],[0,0,1]]
    z = torch.zeros_like(ce); o = torch.ones_like(ce)
    Tec2i = torch.stack([
        torch.stack([ce, -se, z], 1), torch.stack([se, ce, z], 1),
        torch.stack([z, z, o], 1)], 1)
    Tl2i = torch.bmm(Tec2i, Tl2ec)
    Ti2b = _quat_to_T(q)
    Tl2b = torch.bmm(Ti2b, Tl2i)
    # mat_to_euler(Tl2b) (geocentric NED euler)
    d02 = Tl2b[:, 0, 2].clamp(-1.0, 1.0)
    theta = torch.asin(-d02)
    phi = torch.atan2(Tl2b[:, 1, 2], Tl2b[:, 2, 2])
    psi = torch.atan2(Tl2b[:, 0, 1], Tl2b[:, 0, 0])
    psi = torch.where(psi < 0.0, psi + 2 * math.pi, psi)
    euler = torch.stack([phi, theta, psi], 1)
    # vUVW = Ti2b·(eci_vel - omega×eci_pos),  omega×eci_pos = (-ROT·py, ROT·px, 0)
    dvx = vel[:, 0] - (-ROTATION_RATE * py)
    dvy = vel[:, 1] - (ROTATION_RATE * px)
    dvz = vel[:, 2]
    dv = torch.stack([dvx, dvy, dvz], 1).unsqueeze(2)       # (nac,3,1)
    vUVW = torch.bmm(Ti2b, dv).squeeze(2)
    ecef_m = torch.stack([ex, ey, ez], 1) * FT2M
    alt_asl_ft = radius - SEA_LEVEL_RADIUS_FT
    return ecef_m, euler, vUVW, alt_asl_ft


def ned_to_geodetic_np(n, e, d):
    """Official Cartesian N/E/D -> JSBSim seed latitude/longitude/MSL.

    ``pymap3d`` supplies the horizontal inverse.  Its tangent-derived height
    is intentionally discarded; the official vertical contract is MSL=-D.
    """
    import pymap3d as pm
    lat, lon, _ = pm.ned2geodetic(n, e, d, ORIGIN_LAT_DEG, ORIGIN_LON_DEG, ORIGIN_ALT_M)
    return float(lat), float(lon), float(-d) * M2FT


class GpuDogfightVecEnv:
    """벡터화 1v1 dogfight RL env. nenv 개 전투(각 2기: ownship=2e, opponent=2e+1)."""

    OBS_SIZE = OBS_SIZE

    # 2026-09-03 (rule change): the competition now asks for two separate
    # submissions -- one trained for the 3-9 (abeam, perpendicular headings)
    # initial condition and one for head-on. "mixed" keeps the original 3:1
    # blend for reference/back-compat. This selects the *training* reset
    # distribution AND the league evaluation bank together; a scenario-locked
    # model must never be admitted on off-distribution games.
    SCENARIO_HEADON_PROBABILITY = {"three_nine": 0.0, "headon": 1.0, "mixed": 0.25}

    def __init__(self, nenv, substeps=6, precision="fp64", block=128, seed=None,
                 device="cuda", min_altitude_m=HARD_DECK_M, max_engage_time_s=200.0,
                 reward_cfg=None, reward_mode=0, alt_hunt_coef=5.0,
                 scenario="mixed"):
        if precision != "fp64":
            raise ValueError(
                "GpuDogfightVecEnv supports precision='fp64' only: fused observation/reward "
                "kernels require float64 state and action buffers. "
                "Use GpuDogfight directly for standalone FP32 FDM.")
        # step_ratio=6 (대회): RL step 1회 = sim 6프레임 = action_repeat.
        self.sim = GpuDogfight(nenv, substeps=substeps, planes_per_env=2,
                               precision=precision, block=block)
        self.nenv = nenv
        self.nac = self.sim.nac
        self.substeps = substeps
        self.device = device
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)
        # 대회 IC 분포(env_utils.STANDARD_ENV_CONFIG)
        self.center_n = 3500.0
        self.center_e = 0.0
        self.alt_ft_range = (2000.0, 30000.0)
        self.speed_mps_range = (200.0, 300.0)
        self.distA_ft_choices = (2000.0, 2500.0, 3000.0)
        self.dist_headon_ft = 10000.0
        # 시나리오 B(head-on 마주봄) : A(3-9, 수직·반대). "mixed" 는 대회 옛
        # 규정의 1:3(B 확률 0.25), "three_nine"/"headon" 은 새 규정의 단일
        # 초기조건 전용 모델용으로 한쪽만 뽑는다.
        if scenario not in self.SCENARIO_HEADON_PROBABILITY:
            raise ValueError(
                f"unknown scenario {scenario!r}; expected one of "
                f"{sorted(self.SCENARIO_HEADON_PROBABILITY)}")
        self.scenario = str(scenario)
        self.scenario_b_prob = self.SCENARIO_HEADON_PROBABILITY[self.scenario]
        # Official competition hard deck: exactly 1000ft = 304.8m MSL.
        self.min_altitude_m = min_altitude_m
        self.max_engage_time_s = max_engage_time_s
        self.reward_cfg = dict(MY_REWARD_CONFIG if reward_cfg is None else reward_cfg)
        self.reward_mode = validate_reward_mode(reward_mode, alt_hunt_coef)
        self.alt_hunt_coef = float(alt_hunt_coef)
        # 관측(claude164r)/보상(my_reward) 배치 계산기 + 재구성 상태.
        self.obr = BatchObsReward(nenv, device=device)
        # autoreset 용 GPU IC 풀: 재시드를 CPU seed 빌드 없이 GPU gather 로(동기화 제거).
        self.ic_pool = None
        self.ic_pool_size = 4096
        self._evaluation_seed_bank = None
        self._evaluation_seed_bank_key = None
        self._evaluation_seed_metadata = None

    # ── IC 샘플링 ────────────────────────────────────────────────────────────
    def _sample_ic_pair(self, *, alt_ft=None, speed=None, side_swap=None,
                        scenario_b=None, head_swap=None, dist_ft=None):
        """한 env 의 (ownship, target) IC dict 쌍을 대회 분포에서 뽑는다.
        각 dict = build_seed_vector kwargs. 위치는 NED→geodetic 변환."""
        rng = self.rng
        alt_ft = float(rng.uniform(*self.alt_ft_range) if alt_ft is None else alt_ft)
        speed = float(rng.uniform(*self.speed_mps_range) if speed is None else speed)
        side_swap = bool(rng.integers(0, 2) if side_swap is None else side_swap)
        scenario_b = bool(rng.random() < self.scenario_b_prob
                          if scenario_b is None else scenario_b)
        if scenario_b:
            # B: head-on
            half = 0.5 * self.dist_headon_ft * FT2M
            own_n = self.center_n + (-half if side_swap else half)
            tgt_n = self.center_n + (half if side_swap else -half)
            own_hdg = 0.0 if side_swap else 180.0
            tgt_hdg = 180.0 if side_swap else 0.0
        else:
            head_swap = bool(rng.integers(0, 2) if head_swap is None else head_swap)
            dist_ft = float(rng.choice(np.asarray(self.distA_ft_choices))
                            if dist_ft is None else dist_ft)
            half = 0.5 * dist_ft * FT2M
            own_n = self.center_n + (-half if side_swap else half)
            tgt_n = self.center_n + (half if side_swap else -half)
            own_hdg = 270.0 if head_swap else 90.0
            tgt_hdg = 90.0 if head_swap else 270.0
        d = -alt_ft * FT2M   # NED down 음수 = 고도
        return (self._ic_dict(own_n, self.center_e, d, own_hdg, speed),
                self._ic_dict(tgt_n, self.center_e, d, tgt_hdg, speed))

    def _build_evaluation_seed_bank(self, *, paired=False, seed_block=0):
        """Fixed, exactly stratified clean ICs for league evaluation.

        Training resets retain their stochastic distribution.  Evaluation uses
        the same physical IC bank for both policy orderings: exactly one in four
        lanes is head-on, side/heading are balanced, the three 3-9 distances are
        cycled, and altitude/speed cover their ranges with deterministic
        stratified permutations.  No policy-independent warm-up is applied.
        """
        from cuda_fdm.ic import build_seed_vector
        paired = bool(paired)
        if paired and self.nenv % 8:
            raise ValueError("mirrored 3:1 evaluation requires nenv divisible by 8")
        base_count = self.nenv // 2 if paired else self.nenv
        rng = np.random.default_rng(
            (self.seed + 19_870_331 + int(seed_block) * 1_000_003) % (2**63 - 1))
        altitude_order = rng.permutation(base_count)
        speed_order = rng.permutation(base_count)
        rows = np.zeros((self.nac, STATE_N), dtype=np.float64)
        distance_counts = {str(int(distance)): 0 for distance in self.distA_ft_choices}
        for env_index in range(base_count):
            alt_fraction = (float(altitude_order[env_index]) + 0.5) / base_count
            speed_fraction = (float(speed_order[env_index]) + 0.5) / base_count
            alt_ft = self.alt_ft_range[0] + alt_fraction * (
                self.alt_ft_range[1] - self.alt_ft_range[0])
            speed = self.speed_mps_range[0] + speed_fraction * (
                self.speed_mps_range[1] - self.speed_mps_range[0])
            if self.scenario == "mixed":
                scenario_b = (env_index % 4) == 3
                # Cycle only over three-nine lanes so all three approved
                # distances have equal cardinality (head-on lanes do not
                # consume the cycle).
                a_index = env_index - (env_index + 1) // 4
            else:
                # A scenario-locked submission is evaluated only on its own
                # initial condition: admission, solver and historical-audit
                # games must all come from the distribution the policy is
                # actually trained for, or 25% of every league decision would
                # be measured off-distribution.
                scenario_b = self.scenario == "headon"
                a_index = env_index
            side_swap = ((env_index // 4) % 2) == 1
            head_swap = ((env_index // 8) % 2) == 1
            dist_ft = self.distA_ft_choices[a_index % len(self.distA_ft_choices)]
            if not scenario_b:
                distance_counts[str(int(dist_ft))] += 1
            own, target = self._sample_ic_pair(
                alt_ft=alt_ft, speed=speed, side_swap=side_swap,
                scenario_b=scenario_b, head_swap=head_swap, dist_ft=dist_ft)
            rows[2 * env_index] = build_seed_vector(**own)
            rows[2 * env_index + 1] = build_seed_vector(**target)
        if paired:
            # The second half is byte-identical physical state; the evaluator
            # swaps which policy controls plane 0/1 rather than resampling ICs.
            rows[2 * base_count:] = rows[:2 * base_count]
        if self.scenario == "mixed":
            headon = base_count // 4
        else:
            headon = base_count if self.scenario == "headon" else 0
        self._evaluation_seed_metadata = {
            "paired": paired, "seed_block": int(seed_block),
            "scenario": self.scenario,
            "base_count": int(base_count), "three_nine": int(base_count - headon),
            "headon": int(headon), "three_nine_distance_counts": distance_counts,
        }
        return torch.as_tensor(rows, dtype=self.sim.dtype, device=self.device)

    def reset_evaluation(self, *, paired=False, seed_block=0):
        """Reset to a deterministic stratified bank with no random warm-up."""
        key = (bool(paired), int(seed_block))
        if self._evaluation_seed_bank is None or self._evaluation_seed_bank_key != key:
            self._evaluation_seed_bank = self._build_evaluation_seed_bank(
                paired=paired, seed_block=seed_block)
            self._evaluation_seed_bank_key = key
        self.sim.load_seed(self._evaluation_seed_bank)
        self.obr.reset_all()
        self.obr.kernel_init_reward_state(self.sim.states)
        if self.obr.kernel_ready:
            return self.obr.kernel_build_obs(self.sim.states).view(
                self.nenv, 2, self.OBS_SIZE)
        return self.obr.build_obs(self.state9_flat()).view(self.nenv, 2, self.OBS_SIZE)

    def _ic_dict(self, n, e, d, heading_deg, speed_mps):
        lat, lon, alt_ft = ned_to_geodetic_np(n, e, d)
        return dict(lat_deg=lat, lon_deg=lon, alt_ft=alt_ft,
                    vt_fps=speed_mps * M2FT, psi_deg=heading_deg,
                    phi_deg=0.0, theta_deg=0.0, alpha_deg=0.0, beta_deg=0.0,
                    fuel_lbs=6000.0, throttle=0.8)

    def _build_all_seeds(self):
        """nenv 쌍을 샘플 → (nac,101) seed numpy. even=ownship, odd=target."""
        from cuda_fdm.ic import build_seed_vector
        seeds = np.zeros((self.nac, STATE_N), dtype=np.float64)
        for e in range(self.nenv):
            own, tgt = self._sample_ic_pair()
            seeds[2 * e] = build_seed_vector(**own)
            seeds[2 * e + 1] = build_seed_vector(**tgt)
        return seeds

    def _ensure_ic_pool(self):
        """autoreset 용 GPU IC 풀 (ic_pool_size, 2, 101) 을 1회 CPU 생성 후 업로드.
        step 의 autoreset 은 이 풀에서 GPU gather 로 재시드 → CPU seed 빌드/동기화 제거."""
        if self.ic_pool is not None:
            return
        from cuda_fdm.ic import build_seed_vector
        P = self.ic_pool_size
        pool = np.zeros((P, 2, STATE_N), dtype=np.float64)
        for i in range(P):
            own, tgt = self._sample_ic_pair()
            pool[i, 0] = build_seed_vector(**own)
            pool[i, 1] = build_seed_vector(**tgt)
        self.ic_pool = torch.as_tensor(pool, dtype=self.sim.dtype, device=self.device)

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, stagger=True, max_stagger_steps=256, warmup_action_fn=None):
        """대회 분포에서 IC 를 뽑아 모든 env 리셋.
        stagger=True 면 각 env 를 [0,max_stagger_steps) 랜덤 스텝만큼 미리 진행시켜
        에피소드-위상 분포를 고르게 한다(첫 iteration/재개 편향 제거). 이때 재구성
        상태(hp/연료/시간/pqr/action history)도 그 위상까지 함께 진행된다.
        warmup_action_fn(obs)->actions(nac,4): 워밍업에 쓸 정책(없으면 랜덤).
        반환 obs (nenv,2,OBS_SIZE)."""
        seeds = self._build_all_seeds()
        require_finite(seeds, "initial physical states")
        self.sim.load_seed(seeds)
        self.obr.reset_all()
        self.obr.kernel_init_reward_state(self.sim.states)
        if stagger and max_stagger_steps > 0:
            self._staggered_warmup(max_stagger_steps, warmup_action_fn)
        if self.obr.kernel_ready:
            return self.obr.kernel_build_obs(self.sim.states).view(self.nenv, 2, self.OBS_SIZE)
        return self.obr.build_obs(self.state9_flat()).view(self.nenv, 2, self.OBS_SIZE)

    def _staggered_warmup(self, max_steps, action_fn):
        """각 env 를 랜덤 목표위상 p_e∈[0,max_steps) '유효' 스텝만큼 진행시켜 캡처.
        종료-인지: 워밍업 중 이탈(지면관통/발산)한 env 는 즉시 새 IC 로 재시드하고 age 를
        0 으로 되돌린다 → 캡처 시점의 상태는 항상 '유효 IC 로부터 p_e 스텝' 진행된 것.
        물리 상태와 재구성 상태(hp/시간/pqr/action history)를 함께 캡처해 위상을 일치시킨다.
        한 env 의 두 기체는 같은 위상(같은 sim time)을 공유."""
        targ_env = torch.tensor(self.rng.integers(0, max_steps, size=self.nenv),
                                device=self.device)                 # (nenv,)
        age = torch.zeros(self.nenv, dtype=torch.long, device=self.device)
        captured_mask = torch.zeros(self.nenv, dtype=torch.bool, device=self.device)
        captured = self.sim.states.clone()                          # p_e==0 → IC 그대로
        obr_buf = self.obr.clone_state()                            # 재구성도 p_e==0 캡처
        captured_mask |= (targ_env == 0)
        warmup_finite = torch.ones((), dtype=torch.bool, device=self.device)
        for i in range(max_steps):
            # 랜덤 워밍업이면 obs 불필요 → build 생략(reset 가속). 정책 워밍업만 obs 계산.
            if action_fn is None:
                a = self._random_actions()
            else:
                a = action_fn(self.obr.build_obs(self.state9_flat()))
            self.obr.push_actions(a)
            self.sim.step(a, substeps=self.substeps)
            finite_env = torch.isfinite(self.sim.states).view(self.nenv, -1).all(dim=1)
            warmup_finite.logical_and_((finite_env | captured_mask).all())
            s9 = self.state9_flat()
            self.obr.advance(s9)
            dep = self._departed_envs(s9) & (~captured_mask)        # 아직 캡처 안 된 이탈 env
            if dep.any():
                self._seed_envs(torch.nonzero(dep, as_tuple=False).flatten().tolist())
                self.obr.reset_envs(dep)
                self.obr.kernel_init_reward_state(self.sim.states, dep)
                age[dep] = 0
            age = age + (~captured_mask).long()
            hit = (~captured_mask) & (age == targ_env)
            if hit.any():
                captured[hit.repeat_interleave(2)] = self.sim.states[hit.repeat_interleave(2)]
                self.obr.capture_into(obr_buf, hit)
                captured_mask |= hit
            if bool(captured_mask.all()):
                break
        # 끝까지 캡처 못 한 env(목표위상 과대/반복 이탈): 안전한 새 IC 로 마무리.
        if not bool(captured_mask.all()):
            rest = ~captured_mask
            self._seed_envs(torch.nonzero(rest, as_tuple=False).flatten().tolist())
            self.obr.reset_envs(rest)
            self.obr.kernel_init_reward_state(self.sim.states, rest)
            captured[rest.repeat_interleave(2)] = self.sim.states[rest.repeat_interleave(2)]
            self.obr.capture_into(obr_buf, rest)
        if not bool(warmup_finite):
            raise FloatingPointError("non-finite physical trajectory during staggered initialization")
        self.sim.states.copy_(captured)
        self.obr.restore(obr_buf)
        # Warmup transitions were not collected for learning. Anchor the first
        # reward to the restored phase, not to the pre-warmup IC. Do NOT reset
        # HP/time/action history: those were captured at the same physical phase.
        self.obr.kernel_init_reward_state(self.sim.states)

    def _departed_envs(self, s9_flat=None, floor_m=100.0):
        """이탈 판정(env 단위): 두 기체 중 하나라도 고도<floor 또는 비유한.
        s9_flat (nac,9) 를 주면 재계산하지 않는다(워밍업 성능)."""
        s9 = (self.state9() if s9_flat is None
              else s9_flat.view(self.nenv, 2, 9))
        alt = -s9[:, :, S9["D"]]                                    # 고도 m
        bad = (alt < floor_m) | (~torch.isfinite(s9).all(dim=2))
        return bad.any(dim=1)                                       # (nenv,)

    def _seed_envs(self, env_indices):
        """주어진 env 들을 새 IC 쌍으로 재시드(states 해당 행 갱신). CPU 빌드 후 업로드."""
        if not env_indices:
            return
        from cuda_fdm.ic import build_seed_vector
        rows = np.zeros((len(env_indices) * 2, STATE_N), dtype=np.float64)
        for k, e in enumerate(env_indices):
            own, tgt = self._sample_ic_pair()
            rows[2 * k] = build_seed_vector(**own)
            rows[2 * k + 1] = build_seed_vector(**tgt)
        ac_idx = []
        for e in env_indices:
            ac_idx += [2 * e, 2 * e + 1]
        idx_t = torch.tensor(ac_idx, device=self.device)
        self.sim.states[idx_t] = torch.as_tensor(rows, dtype=self.sim.dtype, device=self.device)

    def _random_actions(self):
        """워밍업용 랜덤 액션: stick∈[-0.2,0.2], throttle∈[0.6,0.9] (departure 억제)."""
        a = torch.empty(self.nac, 4, dtype=torch.float64, device=self.device)
        a[:, 0:3].uniform_(-0.2, 0.2)
        a[:, 3].uniform_(0.6, 0.9)
        return a

    def _to_nac4(self, actions):
        """actions (nenv,2,4) 또는 (nac,4) → (nac,4) 텐서."""
        a = torch.as_tensor(actions, dtype=self.sim.dtype, device=self.device)
        return a.reshape(self.nac, 4)

    # ── termination ───────────────────────────────────────────────────────────
    def _termination(self, s9_flat):
        """종료 판정(env 단위). 반환 (terminated (nenv,), truncated (nenv,)).
        규약(termination.py): 고도<min → terminated, HP<=0 → terminated, 비유한 → terminated,
        SIM_TIME>=max_engage (FP64 오차 허용) → truncated.
        여기서 truncated는 HP로 승패를 정하는 실제 경기 종료다. 임의 rollout 절단이
        아니므로 learner는 이 경계에서 미래 가치를 bootstrap하지 않는다.
        (연료·two-circle guard 는 미적용.)"""
        s9 = s9_flat.view(self.nenv, 2, 9)
        alt = -s9[:, :, S9["D"]]                                    # (nenv,2)
        finite = torch.isfinite(s9).all(dim=2)                      # (nenv,2)
        hp = self.obr.hp.view(self.nenv, 2)
        term = ((alt < self.min_altitude_m).any(dim=1)
                | (hp <= 0.0).any(dim=1)
                | (~finite).all(dim=1) | (~finite).any(dim=1))
        trunc = time_limit_reached(self.obr.t_sec, self.max_engage_time_s) & (~term)
        return term, trunc

    # ── step ─────────────────────────────────────────────────────────────────
    def step_training(self, actions):
        """Internal consumer path: retain terminal outcomes, omit unused obs copy."""
        # Keep one-argument step instrumentation compatible. This environment
        # is synchronous; restore the option even if physics/observation fails.
        previous = getattr(self, "_capture_terminal_obs", True)
        self._capture_terminal_obs = False
        try:
            return self.step(actions)
        finally:
            self._capture_terminal_obs = previous

    def step(self, actions, *, capture_terminal_obs=None):
        """actions (nenv,2,4) 또는 (nac,4): [aileron,elevator,rudder,throttle]∈([-1,1]³,[0,1]).

        반환 (obs, reward, done, info):
          obs    (nenv,2,OBS_SIZE) float32 — done env 는 autoreset 된 새 관측.
          reward (nenv,2) — 이번 step 의 관점별 보상.
          done   (nenv,) bool — terminated|truncated.
          info   dict: terminated/truncated (nenv,), terminal_obs (done env 의 종료 관측).
        """
        if capture_terminal_obs is None:
            capture_terminal_obs = getattr(self, "_capture_terminal_obs", True)
        a = self._to_nac4(actions)
        if self.ic_pool is None:
            self._ensure_ic_pool()
        self.sim.step(a, substeps=self.substeps)
        # Capture before autoreset/sanitized observations can hide invalid physics.
        # Stays on GPU; the learner checks one aggregate at the rollout boundary.
        terminal_state_finite = torch.isfinite(self.sim.states).all(dim=1).view(self.nenv, 2).all(dim=1)
        # 융합 커널: action push + advance + 종료 + reward 를 한 launch 로.
        reward, term_u8, trunc_u8 = self.obr.kernel_advance(
            self.sim.states, a, cfg=self.reward_cfg,
            min_alt=self.min_altitude_m, max_time=self.max_engage_time_s,
            reward_mode=self.reward_mode, alt_hunt_coef=self.alt_hunt_coef)
        term = term_u8.bool(); trunc = trunc_u8.bool()
        done = term | trunc
        reward = reward.view(self.nenv, 2).clone()
        # 종료 관측(autoreset 전) — 융합 build_obs 커널.
        self.obr.kernel_build_obs(self.sim.states)
        terminal_obs = (self.obr.obs_buf.view(self.nenv, 2, self.OBS_SIZE).clone()
                        if capture_terminal_obs else None)
        # 승패 귀속용 per-기체 terminal hp/MSL altitude (autoreset 전 캡처, sync 없음).
        # This exact radial JSBSim ASL is also the kernel termination altitude.
        radius_ft = torch.linalg.vector_norm(self.sim.states[:, 0:3], dim=1)
        terminal_alt_m = ((radius_ft - SEA_LEVEL_RADIUS_FT) * FT2M).view(self.nenv, 2)
        terminal_hp = self.obr.hp.view(self.nenv, 2).clone()
        # autoreset — **완전 동기화 없음**: GPU IC 풀에서 gather 후 done env 만 torch.where 로
        # 덮어쓰고, 재구성은 masked_fill_ 로 초기화. .item()/.any() 없음 → step 이 CPU-GPU 동기화
        # 를 일으키지 않는다(done 은 GPU 텐서로 반환, 소비자가 필요할 때 동기화).
        pick = torch.randint(0, self.ic_pool_size, (self.nenv,), device=self.device)
        new_states = self.ic_pool[pick].reshape(self.nac, STATE_N)
        mac = done.repeat_interleave(2).unsqueeze(1)                # (nac,1)
        self.sim.states.copy_(torch.where(mac, new_states, self.sim.states))
        self.obr.reset_envs(done)
        self.obr.kernel_init_reward_state(self.sim.states, done)
        # Preserve already computed observations for live lanes. No host any()
        # check: the kernel cheaply skips non-reset lanes using the GPU mask.
        self.obr.kernel_build_obs(self.sim.states, done)
        obs = self.obr.obs_buf.view(self.nenv, 2, self.OBS_SIZE)
        info = {"terminated": term, "truncated": trunc, "terminal_obs": terminal_obs,
                "terminal_hp": terminal_hp, "terminal_alt_m": terminal_alt_m, "done": done,
                "terminal_state_finite": terminal_state_finite}
        return obs, reward, done, info

    # ── 상태 export ──────────────────────────────────────────────────────────
    def state9_flat(self):
        """대회 9-DOF 상태 (nac,9): N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s].
        states 에서 자체 계산(obs 의존 없음 → reset 직후에도 유효)."""
        ecef_m, euler, vUVW, alt_asl_ft = kinematics_state(self.sim.states)
        n, e, d = ned_from_ecef_altasl(ecef_m, alt_asl_ft * FT2M)
        ned = torch.stack([n, e, d], dim=1)                  # (nac,3) m
        rpy = euler * R2D                                    # rad→deg
        uvw = vUVW * FT2M                                    # fps→m/s
        return torch.cat([ned, rpy, uvw], dim=1)             # (nac,9)

    def state9(self):
        """state9_flat 을 (nenv,2,9) 로."""
        return self.state9_flat().view(self.nenv, 2, 9)
