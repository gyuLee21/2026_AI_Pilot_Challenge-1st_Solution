# -*- coding: utf-8 -*-
"""claude164r 관측 + my_reward 보상의 GPU 벡터화 (torch, fp64).

CPU 참조(claude_code/my_observation.py, my_reward.py, GeoMathUtil.GeometryInfo)와
**동일 규약**을 재현한다. 레이아웃/상수는 참조 모듈에서 직접 import 해 정합을 강제한다
(참조가 바뀌면 검증 테스트가 어긋난다).

핵심 설계
---------
- 모든 상태를 **기체 단위(nac=2*nenv)** 로 관리한다. 기체 a 의 상대는 partner=a^1.
  거리·시간은 env 안에서 두 기체가 공유한다.
- StateReconstructor 를 배치 텐서로 유지한다(hp/fuel/t_sec/자세 history/pqr/action history
  /geometry·상대 log 고도 tracker). advance() 는 RL step 당 한 번, sim.step **후**
  새 상태로 적분한다.
- 관측은 관점 기체 a(own=a, target=a^1)에서 (nac,OBS_SIZE)로 한 번에 만든다.
- 보상도 관점 기체 a 에서 (nac,)로 만든다(self-play: 두 기체 모두 보상).

state9 입력: (nac,9) = N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s].
"""
import ctypes as _CT
import math
import sys
from pathlib import Path

import torch
from cuda_fdm.future_aux import AUX_FEATURE_DIM
from cuda_fdm.reward_modes import (damage_reward, ALTITUDE_HUNT_REWARD, ALTITUDE_LOG_SCALE_M,
                                   ALTITUDE_LOG_FLOOR, validate_reward_mode)

_RELEASE = Path(__file__).resolve().parents[1]
if str(_RELEASE) not in sys.path:
    sys.path.insert(0, str(_RELEASE))
if str(_RELEASE / "claude_code") not in sys.path:
    sys.path.insert(0, str(_RELEASE / "claude_code"))
if str(_RELEASE / "cuda_fdm" / "tests") not in sys.path:
    sys.path.insert(0, str(_RELEASE / "cuda_fdm" / "tests"))
_GEN = _RELEASE / "cuda_fdm" / "gen"

# CPU 참조 상수/레이아웃 (정합 강제)
import my_observation as R          # claude_code/my_observation.py
import my_reward as RW              # claude_code/my_reward.py

OBS_SIZE = R.OBSERVATION_SIZE       # 214 (=50 scalar + 144 vector[가속도 30 포함] + 20 action-history)
VEC_LAYOUT = R._VEC_LAYOUT          # [(key, kind, frame), ...] 38개
FRAME_NAMES = R.FRAME_NAMES
ACT_LEN = R.ACTION_HISTORY_LEN
ACT_DIM = R.ACTION_DIM

D2R = 3.141592653589793 / 180.0
R2D = 180.0 / 3.141592653589793
_FT2M = R.FEET_TO_METER             # 0.3048
_M2FT = R.METER_TO_FEET             # 3.28084 (obs 규약)
_G = R.G


# ══════════════════════════════════════════════════════════════════════════════
# 배치 기하 (GeometryInfo 벡터화). 모두 (M,9) own/tgt → (M,) 또는 (M,3).
# ══════════════════════════════════════════════════════════════════════════════
def ned_to_body(rpy_deg):
    """(M,3) roll/pitch/yaw[deg] → R_ned_to_body (M,3,3) = Tx@Ty@Tz.
    GeometryInfo/_ned_to_body_matrix 와 동일한 부호 규약."""
    r = rpy_deg[:, 0] * D2R
    p = rpy_deg[:, 1] * D2R
    y = rpy_deg[:, 2] * D2R
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    z = torch.zeros_like(r); o = torch.ones_like(r)
    Tx = torch.stack([torch.stack([o, z, z], 1),
                      torch.stack([z, cr, sr], 1),
                      torch.stack([z, -sr, cr], 1)], 1)
    Ty = torch.stack([torch.stack([cp, z, -sp], 1),
                      torch.stack([z, o, z], 1),
                      torch.stack([sp, z, cp], 1)], 1)
    Tz = torch.stack([torch.stack([cy, sy, z], 1),
                      torch.stack([-sy, cy, z], 1),
                      torch.stack([z, z, o], 1)], 1)
    return torch.bmm(Tx, torch.bmm(Ty, Tz))


def _mv(R3, v):
    """batched (M,3,3)·(M,3) → (M,3)."""
    return torch.einsum('mij,mj->mi', R3, v)


def _unit(v, eps=0.0):
    """(M,3) 정규화. norm==0 이면 원본(참조 규약: 0 이면 그대로)."""
    n = torch.linalg.norm(v, dim=1, keepdim=True)
    return torch.where(n > 0, v / n.clamp_min(1e-300), v)


def distance_m(own9, tgt9):
    return torch.linalg.norm(tgt9[:, :3] - own9[:, :3], dim=1)


def ata_deg(own9, tgt9):
    """3D antenna train angle(own→tgt), 0~180 (부호 없음). _get_antenna_train_angle proj=False."""
    pu = _unit(tgt9[:, :3] - own9[:, :3])
    Rnb = ned_to_body(own9[:, 3:6])
    pt = _mv(Rnb, pu)
    return torch.arccos(pt[:, 0].clamp(-1.0, 1.0)) * R2D


def aspect_deg(own9, tgt9):
    """3D aspect angle, 부호 있음. _get_aspect_angle proj=False."""
    Rt = ned_to_body(tgt9[:, 3:6])
    pu = _unit(own9[:, :3] - tgt9[:, :3])
    b = _mv(Rt, pu)                     # Tx@Ty@Tz @ p_unit
    pt0 = -b[:, 0]; pt1 = -b[:, 1]; pt2 = b[:, 2]   # Tz_pi = diag(-1,-1,1)
    sign = torch.where(pt1 > 0, torch.ones_like(pt1),
                       torch.where(pt1 < 0, -torch.ones_like(pt1),
                                   torch.where(pt2 >= 0, torch.ones_like(pt1),
                                               -torch.ones_like(pt1))))
    return sign * torch.arccos(pt0.clamp(-1.0, 1.0)) * R2D


def los_az_el_deg(own9, tgt9):
    """LOS azimuth(-180~180), elevation(-90~90) in own body. _get_los_angle."""
    du = _unit(tgt9[:, :3] - own9[:, :3])
    Rnb = ned_to_body(own9[:, 3:6])
    db = _mv(Rnb, du)
    az = torch.atan2(db[:, 1], db[:, 0]) * R2D
    el = -torch.arcsin(db[:, 2].clamp(-1.0, 1.0)) * R2D
    return az, el


# ══════════════════════════════════════════════════════════════════════════════
# 방향-정렬 좌표계 / 뱅크각 / SO(3) log (my_observation 벡터화)
# ══════════════════════════════════════════════════════════════════════════════
def dir_frame(x_ned):
    """(M,3) x축 방향 → R_ned_to_frame (M,3,3), rows=[x,y,z]. roll 은 중력 고정.
    x∥중력이면 north→east fallback. |x|<1e-8 이면 항등. _dir_frame 규약."""
    M = x_ned.shape[0]
    dev, dt = x_ned.device, x_ned.dtype
    eye = torch.eye(3, device=dev, dtype=dt).expand(M, 3, 3)
    nx = torch.linalg.norm(x_ned, dim=1, keepdim=True)
    safe = (nx.squeeze(1) >= 1e-8)
    xn = x_ned / nx.clamp_min(1e-300)

    def proj_out(ref):   # ref (3,) → z = ref - (ref·xn)xn  (M,3)
        r = ref.view(1, 3)
        d = (xn * r).sum(1, keepdim=True)
        return r - d * xn

    down = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt)
    north = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt)
    east = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt)
    z_d = proj_out(down);  n_d = torch.linalg.norm(z_d, dim=1)
    z_n = proj_out(north); n_n = torch.linalg.norm(z_n, dim=1)
    z_e = proj_out(east)
    use_d = (n_d >= 1e-6)
    use_n = (~use_d) & (n_n >= 1e-6)
    z = torch.where(use_d.unsqueeze(1), z_d,
                    torch.where(use_n.unsqueeze(1), z_n, z_e))
    z = z / torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(1e-300)
    y = torch.linalg.cross(z, xn, dim=1)
    y = y / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-300)
    z = torch.linalg.cross(xn, y, dim=1)
    Rf = torch.stack([xn, y, z], 1)                 # rows
    return torch.where(safe.view(M, 1, 1), Rf, eye)


def bank_sincos(Rb2n, dir_ned):
    """방향벡터 축 뱅크각 μ 의 (sin,cos). _bank_about_dir 벡터화.
    Rb2n=(M,3,3) body→ned. body y축(ned)=Rb2n[:,:,1]."""
    Rf = dir_frame(dir_ned)
    body_y_ned = Rb2n[:, :, 1]
    yv = _mv(Rf, body_y_ned)
    mu = torch.atan2(yv[:, 2], yv[:, 1])
    return torch.sin(mu), torch.cos(mu)


def log_so3(Rm):
    """(M,3,3) 회전행렬 → axis*angle (M,3). _log_so3 벡터화(θ<1e-8 → 0)."""
    tr = Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2]
    cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_t)
    ax = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2],
                      Rm[:, 0, 2] - Rm[:, 2, 0],
                      Rm[:, 1, 0] - Rm[:, 0, 1]], 1)
    denom = 2.0 * torch.sin(theta)
    ok = (theta >= 1e-8) & (denom.abs() >= 1e-8)
    # denom 이 음수일 수 있으니 안전 나눗셈(0 근처는 1 로 대체 후 where 로 무효화).
    safe_denom = torch.where(denom.abs() < 1e-300, torch.ones_like(denom), denom)
    scale = torch.where(ok, theta / safe_denom, torch.zeros_like(theta))
    return ax * scale.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# 스칼라 helper
# ══════════════════════════════════════════════════════════════════════════════
def normalize_t(x, lo, hi):
    """observation.normalize 벡터화: clip 후 [-1,1] 선형."""
    if hi <= lo:
        return torch.zeros_like(x)
    mid = (hi + lo) * 0.5
    half = (hi - lo) * 0.5
    return (x.clamp(lo, hi) - mid) / half


def _sincos_deg(a_deg):
    r = a_deg * D2R
    return torch.sin(r), torch.cos(r)


def damage_rate_t(r_ft, ata_deg_abs, t_sec):
    """cone-damage rate 벡터화. my_observation.damage_rate 와 동일(tier1>2>3 우선)."""
    a = ata_deg_abs.abs()
    z = torch.zeros_like(r_ft)
    m1 = (r_ft >= R.MIN_DAMAGE_RANGE_FT) & (r_ft <= R.TIER1_MAX_RANGE_FT) & (a < R.TIER1_CONE_DEG)
    v1 = 1.0 * (R.TIER1_MAX_RANGE_FT - r_ft) / (R.TIER1_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    m2 = (t_sec >= R.TIER2_START_SEC) & (r_ft >= R.MIN_DAMAGE_RANGE_FT) & \
         (r_ft <= R.TIER2_MAX_RANGE_FT) & (a < R.TIER2_CONE_DEG)
    v2 = 0.3 * (R.TIER2_MAX_RANGE_FT - r_ft) / (R.TIER2_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    m3 = (t_sec >= R.TIER3_START_SEC) & (r_ft >= R.MIN_DAMAGE_RANGE_FT) & \
         (r_ft <= R.TIER3_MAX_RANGE_FT) & (a < R.TIER3_CONE_DEG)
    v3 = 0.1 * (R.TIER3_MAX_RANGE_FT - r_ft) / (R.TIER3_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    return torch.where(m1, v1, torch.where(m2, v2, torch.where(m3, v3, z)))


def _active_envelope(t_sec):
    """(cone_deg, max_range_ft) 배치. _active_damage_envelope."""
    cone = torch.where(t_sec >= R.TIER3_START_SEC, torch.full_like(t_sec, R.TIER3_CONE_DEG),
                       torch.where(t_sec >= R.TIER2_START_SEC, torch.full_like(t_sec, R.TIER2_CONE_DEG),
                                   torch.full_like(t_sec, R.TIER1_CONE_DEG)))
    rng = torch.where(t_sec >= R.TIER3_START_SEC, torch.full_like(t_sec, R.TIER3_MAX_RANGE_FT),
                      torch.where(t_sec >= R.TIER2_START_SEC, torch.full_like(t_sec, R.TIER2_MAX_RANGE_FT),
                                  torch.full_like(t_sec, R.TIER1_MAX_RANGE_FT)))
    return cone, rng


# ── final-safe reward geometry (torch reference; fused kernel mirrors this) ──
def _smoothstep_t(x, edge0, edge1):
    z = ((x - float(edge0)) / (float(edge1) - float(edge0))).clamp(0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def _velocity_ned_t(s9):
    return _mv(ned_to_body(s9[:, 3:6]).transpose(1, 2), s9[:, 6:9])


def _unit_fallback_t(v, fallback):
    n = torch.linalg.norm(v, dim=1, keepdim=True)
    fn = torch.linalg.norm(fallback, dim=1, keepdim=True)
    fallback_u = fallback / fn.clamp_min(1e-9)
    return torch.where(n >= 1e-9, v / n.clamp_min(1e-9), fallback_u)


def _alignment_t(a, b, power=3.0):
    au = _unit_fallback_t(a, torch.tensor([1.0, 0.0, 0.0], device=a.device,
                                          dtype=a.dtype).expand_as(a))
    bu = _unit_fallback_t(b, au)
    q = ((1.0 + (au * bu).sum(1).clamp(-1.0, 1.0)) * 0.5)
    return q.pow(float(power)).clamp(0.0, 1.0)


def final_safe_geometry(own, tgt, t_sec, cfg):
    """각 관점 기체의 S와 대칭 G=S-own-S-enemy를 GPU tensor로 계산한다."""
    rel = tgt[:, :3] - own[:, :3]
    dist_m = torch.linalg.norm(rel, dim=1)
    dist_ft = dist_m / _FT2M
    los = _unit_fallback_t(rel, torch.tensor([1.0, 0.0, 0.0], device=own.device,
                                             dtype=own.dtype).expand_as(rel))
    ata = ata_deg(own, tgt).abs()

    Rnb = ned_to_body(own[:, 3:6])
    forward = Rnb[:, 0, :]  # body x-axis expressed in NED
    own_v = _velocity_ned_t(own)
    tgt_v = _velocity_ned_t(tgt)
    own_speed = torch.linalg.norm(own_v, dim=1).clamp_min(float(cfg["lead_speed_floor_mps"]))
    tau = (dist_m / own_speed).clamp(float(cfg["lead_prediction_min_sec"]),
                                     float(cfg["lead_prediction_max_sec"]))
    lead_dir = _unit_fallback_t(tgt[:, :3] + tgt_v * tau.unsqueeze(1) - own[:, :3], los)
    path = _unit_fallback_t(own_v, forward)
    q_nose = _alignment_t(forward, lead_dir, cfg["alignment_power"])
    q_path = _alignment_t(path, lead_dir, cfg["alignment_power"])
    intercept = (float(cfg["lead_nose_weight"]) * q_nose
                 + float(cfg["lead_path_weight"]) * q_path)
    intercept /= float(cfg["lead_nose_weight"]) + float(cfg["lead_path_weight"])
    direct = _alignment_t(forward, los, cfg["alignment_power"])
    lead_weight = (_smoothstep_t(dist_ft, cfg["lead_direct_range_ft"],
                                 cfg["lead_full_range_ft"])
                   * _smoothstep_t(ata, cfg["lead_direct_ata_deg"],
                                   cfg["lead_full_ata_deg"]))
    safe_lead = lead_weight * intercept + (1.0 - lead_weight) * direct

    full = float(cfg["broad_ata_full_control_angle_deg"])
    broad = torch.where(ata <= full, torch.ones_like(ata),
                        ((180.0 - ata) / (180.0 - full)).clamp(0.0, 1.0))
    angle_gate = 1.0 - _smoothstep_t(
        ata, cfg["weapon_full_alignment_deg"], cfg["weapon_transition_angle_deg"])
    range_gate = (_smoothstep_t(dist_ft, cfg["control_zone_near_zero_ft"],
                                cfg["control_zone_near_full_ft"])
                  * (1.0 - _smoothstep_t(dist_ft, cfg["control_zone_far_full_ft"],
                                         cfg["control_zone_far_zero_ft"])))
    closing = (own_v - tgt_v).mul(los).sum(1)
    mean_speed = 0.5 * (torch.linalg.norm(own_v, dim=1)
                        + torch.linalg.norm(tgt_v, dim=1))
    closing_ratio = closing / mean_speed.clamp_min(float(cfg["control_zone_speed_floor_mps"]))
    desired = (float(cfg["control_zone_far_closure_ratio"])
               * _smoothstep_t(dist_ft, cfg["control_zone_far_closure_start_ft"],
                               cfg["control_zone_far_closure_full_ft"])
               + float(cfg["control_zone_close_opening_ratio"])
               * (1.0 - _smoothstep_t(dist_ft,
                                      cfg["control_zone_close_opening_full_ft"],
                                      cfg["control_zone_close_opening_zero_ft"])))
    closure_q = torch.exp(-0.5 * ((closing_ratio - desired)
                                  / float(cfg["control_zone_closure_sigma"])) ** 2)
    control_zone = (angle_gate * range_gate * closure_q).clamp(0.0, 1.0)

    fine = damage_rate_t(dist_ft, ata, t_sec)
    side = (0.50 * broad + 0.20 * safe_lead + 0.20 * control_zone + 0.10 * fine)
    return side.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 배치 reconstructor + obs/reward
# ══════════════════════════════════════════════════════════════════════════════
class BatchObsReward:
    """nac=2*nenv 기체의 관측/보상 배치 계산기 + 재구성 상태 보유."""

    def __init__(self, nenv, device="cuda", dtype=torch.float64, enable_kernel=True):
        self.nenv = nenv
        self.nac = 2 * nenv
        self.device = device
        self.dtype = dtype
        self.dt = R.DT_PER_STEP
        idx = torch.arange(self.nac, device=device)
        self.partner = idx ^ 1                       # (nac,) 상대 기체 index
        self.env_of = idx // 2                        # (nac,) → env
        self._alloc()
        self.reset_all()
        # ── 융합 NVRTC 커널(advance+reward, build_obs) — step() 핫패스 가속 ──
        self.kernel_ready = False
        if enable_kernel and dtype == torch.float64:
            self._init_kernels()

    def _alloc(self):
        n, ne, dev, dt = self.nac, self.nenv, self.device, self.dtype
        self.hp = torch.ones(n, device=dev, dtype=dt)
        self.fuel = torch.ones(n, device=dev, dtype=dt)
        self.t_sec = torch.zeros(ne, device=dev, dtype=dt)
        self.prev_att = torch.zeros(n, 3, device=dev, dtype=dt)
        self.prev_valid = torch.zeros(n, device=dev, dtype=torch.bool)
        self.pqr = torch.zeros(n, 3, device=dev, dtype=dt)
        # 선가속도: NED 속도의 step 차분. pqr 과 같은 추정 규약이라 prev_valid 를 공유한다
        # (첫 step 은 직전 값이 없어 0). accel 은 NED 로 저장한다 — pqr 이 body 로 저장돼
        # 관측 시점에 NED 로 도는 것과 달리, 배포측 StateReconstructor 가 NED 로 만든다.
        self.prev_vel_ned = torch.zeros(n, 3, device=dev, dtype=dt)
        self.accel = torch.zeros(n, 3, device=dev, dtype=dt)
        self.last_dmg_dealt = torch.zeros(n, device=dev, dtype=dt)   # rate
        self.last_dmg_taken = torch.zeros(n, device=dev, dtype=dt)   # rate
        self.hp_loss = torch.zeros(n, device=dev, dtype=dt)          # 이번 step a 의 HP 손실
        self.act_hist = torch.zeros(n, ACT_LEN, ACT_DIM, device=dev, dtype=dt)
        self.prev_x = torch.zeros(n, device=dev, dtype=dt)           # 직전 geometry advantage
        self.prev_x_valid = torch.zeros(n, device=dev, dtype=torch.bool)
        self.prev_alt_log = torch.zeros(n, device=dev, dtype=dt)      # own log altitude; partner supplies hunter target
        self.aux_features = None  # training labels only; never part of the 184D observation

    def enable_aux_capture(self):
        if self.aux_features is None:
            self.aux_features = torch.zeros(self.nenv, AUX_FEATURE_DIM,
                                             dtype=torch.float64, device=self.device)

    # ── 융합 커널 (advance+reward, build_obs) ─────────────────────────────────
    def _init_kernels(self):
        import cuda_rt
        src = (_GEN / "obs_kernel.cu").read_text(encoding="utf-8")
        cap = torch.cuda.get_device_capability()
        arch = f"compute_{cap[0]}{cap[1]}"
        ptx = cuda_rt.compile_ptx(src, arch)
        self._k_adv = cuda_rt.Kernel(ptx, "advance_kernel")
        self._k_init_reward = cuda_rt.Kernel(ptx, "init_reward_kernel")
        self._k_obs = cuda_rt.Kernel(ptx, "build_obs_kernel")
        self.block = 128
        self.obs_buf = torch.zeros(self.nac, OBS_SIZE, dtype=torch.float32, device=self.device)
        self.reward_buf = torch.zeros(self.nac, dtype=torch.float64, device=self.device)
        self.term_buf = torch.zeros(self.nenv, dtype=torch.uint8, device=self.device)
        self.trunc_buf = torch.zeros(self.nenv, dtype=torch.uint8, device=self.device)
        self._all_env_mask = torch.ones(self.nenv, dtype=torch.uint8, device=self.device)
        # Official BaseLat/BaseLon origin (must match rl_env exactly).
        olat = math.radians(37.240778); olon = math.radians(131.869556)
        slat, clat = math.sin(olat), math.cos(olat)
        slon, clon = math.sin(olon), math.cos(olon)
        A, E2 = 6378137.0, 0.0066943799901411
        Nr = A / math.sqrt(1.0 - E2 * slat * slat)
        self._ox, self._oy, self._oz = Nr * clat * clon, Nr * clat * slon, Nr * (1.0 - E2) * slat
        self._osc = (slat, clat, slon, clon)
        self.kernel_ready = True

    def _origin_args(self):
        s, c, so, co = self._osc
        return [_CT.c_double(self._ox), _CT.c_double(self._oy), _CT.c_double(self._oz),
                _CT.c_double(s), _CT.c_double(c), _CT.c_double(so), _CT.c_double(co)]

    def _kernel_input(self, tensor, shape, dtype, label):
        """Validate the raw-pointer ABI before a driver launch can read memory."""
        if not torch.is_tensor(tensor) or tensor.dtype not in dtype:
            raise TypeError(f"{label} kernel requires dtype {dtype}; no implicit pointer cast")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{label} shape {tuple(tensor.shape)} != {tuple(shape)}")
        if tensor.device.type != "cuda" or tensor.device != self.hp.device:
            raise ValueError(f"{label} must be on the reconstructor's CUDA device")
        if not self.kernel_ready:
            raise ValueError("FP64 CUDA observation/reward kernels are not initialized")
        return tensor.contiguous()

    def _kernel_states(self, states):
        return self._kernel_input(states, (self.nac, 101), (torch.float64,), "FDM states")

    def _kernel_mask(self, mask):
        value = self._all_env_mask if mask is None else mask
        return self._kernel_input(value, (self.nenv,), (torch.bool, torch.uint8), "environment mask")

    def kernel_advance(self, states, actions, cfg=None, min_alt=304.8, max_time=200.0,
                       reward_mode=0, alt_hunt_coef=5.0):
        """융합 advance+reward 커널(1 thread/env): action push, hp/연료/pqr/시간 적분,
        종료 판정, 보상 계산을 한 번에. sim.step **후** 호출. 반환 (reward(nac,), term, trunc)."""
        cfg = cfg if cfg is not None else RW.MY_REWARD_CONFIG
        mode = validate_reward_mode(reward_mode, alt_hunt_coef)
        st = self._kernel_states(states)
        ac = self._kernel_input(actions, (self.nac, 4), (torch.float64,), "actions")
        own_w = float(cfg.get("own_damage_weight", 0.5))
        p = lambda t: _CT.c_void_p(t.data_ptr())
        tensors = (st, ac, self.hp, self.fuel, self.t_sec, self.prev_att,
                   self.prev_valid, self.pqr, self.prev_vel_ned, self.accel,
                   self.last_dmg_dealt, self.last_dmg_taken,
                   self.hp_loss, self.act_hist, self.prev_x, self.prev_x_valid,
                   self.prev_alt_log,
                   self.reward_buf, self.term_buf, self.trunc_buf)
        args = [p(t) for t in tensors] + [_CT.c_int(self.nenv)]
        args += self._origin_args()
        args += [_CT.c_double(self.dt), _CT.c_double(min_alt), _CT.c_double(max_time),
                 _CT.c_double(own_w), _CT.c_double(float(cfg["damage_scale"])),
                 _CT.c_double(float(cfg.get("altitude_settlement_scale", cfg["damage_scale"]))),
                 _CT.c_double(float(cfg["shaping_reward_scale"])),
                  _CT.c_double(float(cfg["win_reward"])), _CT.c_double(float(cfg["loss_reward"])),
                  _CT.c_double(float(cfg["timeout_win_reward"])),
                  _CT.c_double(float(cfg["timeout_loss_reward"])),
                  _CT.c_double(float(cfg["timeout_draw_reward"])),
                  _CT.c_int(mode), _CT.c_double(float(alt_hunt_coef)),
                  _CT.c_int({"remaining_hp": 0, "result": 1, "result_remaining_hp": 2}[cfg.get("altitude_terminal_mode", "remaining_hp")]),
                  _CT.c_double(float(cfg.get("altitude_win_reward", cfg["win_reward"]))),
                  _CT.c_double(float(cfg.get("altitude_loss_reward", cfg["loss_reward"])))]
        grid = ((self.nenv + self.block - 1) // self.block, 1, 1)
        self._k_adv.launch(grid, (self.block, 1, 1), args, tensors=tensors)
        return self.reward_buf, self.term_buf, self.trunc_buf

    def kernel_init_reward_state(self, states, env_mask=None):
        """현재 물리 상태/시각으로 geometry·고도 기준값만 초기화한다."""
        st = self._kernel_states(states)
        mask = self._kernel_mask(env_mask)
        p = lambda t: _CT.c_void_p(t.data_ptr())
        tensors = (st, mask, self.prev_x, self.prev_x_valid, self.prev_alt_log, self.t_sec)
        args = [p(t) for t in tensors] + [_CT.c_int(self.nenv)]
        args += self._origin_args()
        grid = ((self.nenv + self.block - 1) // self.block, 1, 1)
        self._k_init_reward.launch(grid, (self.block, 1, 1), args, tensors=tensors)

    def kernel_build_obs(self, states, env_mask=None):
        """융합 build_obs 커널(1 thread/기체): states+재구성 → obs(nac,214) f32. 순수(비파괴)."""
        st = self._kernel_states(states)
        mask = self._kernel_mask(env_mask)
        p = lambda t: _CT.c_void_p(t.data_ptr())
        tensors = (st, mask, self.hp, self.fuel, self.t_sec, self.pqr, self.accel,
                   self.last_dmg_dealt, self.last_dmg_taken, self.act_hist, self.obs_buf)
        args = [p(t) for t in tensors] + [_CT.c_int(self.nac)]
        args += self._origin_args()
        args.append(p(self.aux_features) if self.aux_features is not None else _CT.c_void_p())
        if self.aux_features is not None:
            tensors += (self.aux_features,)
        grid = ((self.nac + self.block - 1) // self.block, 1, 1)
        self._k_obs.launch(grid, (self.block, 1, 1), args, tensors=tensors)
        return self.obs_buf

    # ── 상태 리셋 (env 단위) ──────────────────────────────────────────────────
    def reset_all(self):
        self.hp.fill_(1.0); self.fuel.fill_(1.0); self.t_sec.zero_()
        self.prev_att.zero_(); self.prev_valid.zero_(); self.pqr.zero_()
        self.prev_vel_ned.zero_(); self.accel.zero_()
        self.last_dmg_dealt.zero_(); self.last_dmg_taken.zero_(); self.hp_loss.zero_()
        self.act_hist.zero_(); self.prev_x.zero_(); self.prev_x_valid.zero_()
        self.prev_alt_log.zero_()

    def reset_envs(self, env_mask):
        """env_mask (nenv,) bool 인 env 의 재구성 상태 초기화(autoreset/stagger reseed).
        masked_fill_ 로 **in-place·동기화없음**(GPU 스칼라 카운트 불필요) — 커널 경로에서
        텐서 주소가 유지돼 매 step 재할당/동기화 오버헤드를 없앤다."""
        ac = env_mask.repeat_interleave(2)           # (nac,)
        ac1 = ac.unsqueeze(1)
        self.hp.masked_fill_(ac, 1.0)
        self.fuel.masked_fill_(ac, 1.0)
        self.t_sec.masked_fill_(env_mask, 0.0)
        self.prev_att.masked_fill_(ac1, 0.0)
        self.prev_valid.masked_fill_(ac, False)
        self.pqr.masked_fill_(ac1, 0.0)
        self.prev_vel_ned.masked_fill_(ac1, 0.0)
        self.accel.masked_fill_(ac1, 0.0)
        self.last_dmg_dealt.masked_fill_(ac, 0.0)
        self.last_dmg_taken.masked_fill_(ac, 0.0)
        self.hp_loss.masked_fill_(ac, 0.0)
        self.act_hist.masked_fill_(ac.view(-1, 1, 1), 0.0)
        self.prev_x.masked_fill_(ac, 0.0)
        self.prev_x_valid.masked_fill_(ac, False)
        self.prev_alt_log.masked_fill_(ac, 0.0)

    # ── stagger 용 상태 snapshot/capture/restore ──────────────────────────────
    _AC_KEYS = ("hp", "fuel", "prev_att", "prev_valid", "pqr", "prev_vel_ned",
                "accel", "last_dmg_dealt", "last_dmg_taken", "hp_loss", "act_hist",
                "prev_x", "prev_x_valid", "prev_alt_log")
    _ENV_KEYS = ("t_sec",)

    def clone_state(self):
        buf = {k: getattr(self, k).clone() for k in self._AC_KEYS + self._ENV_KEYS}
        if self.aux_features is not None:
            buf["aux_features"] = self.aux_features.clone()
        return buf

    def capture_into(self, buf, env_mask):
        """env_mask 인 env 의 현재 상태를 buf 로 복사(그 위상에서 캡처)."""
        ac = env_mask.repeat_interleave(2)
        for k in self._AC_KEYS:
            buf[k][ac] = getattr(self, k)[ac]
        for k in self._ENV_KEYS:
            buf[k][env_mask] = getattr(self, k)[env_mask]
        if self.aux_features is not None:
            buf["aux_features"][env_mask] = self.aux_features[env_mask]

    def restore(self, buf):
        for k in self._AC_KEYS + self._ENV_KEYS:
            if k not in buf:
                raise KeyError(f"obs_reward stagger snapshot missing required key {k!r}")
            getattr(self, k).copy_(buf[k])
        if self.aux_features is not None:
            self.aux_features.copy_(buf["aux_features"])

    # ── action history push (RL step 당 한 번, 다음 obs 가 최근 action 포함) ────
    def push_actions(self, actions_nac4):
        a = actions_nac4.to(self.dtype)
        self.act_hist = torch.roll(self.act_hist, 1, dims=1)
        self.act_hist[:, 0, :] = a[:, :ACT_DIM]

    # ── advance: sim.step 후 새 상태로 재구성 적분 ────────────────────────────
    def advance(self, states9):
        """states9 (nac,9). hp/fuel/t_sec/pqr/last_dmg 갱신. build_obs/reward 전에 호출."""
        own = states9
        tgt = states9[self.partner]
        r_m = distance_m(own, tgt)
        r_ft = r_m * _M2FT                                   # obs 규약(*3.28084)
        ata_a = ata_deg(own, tgt)                            # a→partner, 0..180
        t_ac = self.t_sec[self.env_of]                       # (nac,)
        rate_deals = damage_rate_t(r_ft, ata_a, t_ac)        # a 가 partner 에 가하는 rate
        rate_taken = rate_deals[self.partner]                # a 가 받는 rate
        hp_before = self.hp
        hp_after = (hp_before - rate_taken * self.dt).clamp_min(0.0)
        self.hp_loss = hp_before - hp_after                  # 이번 step a 의 HP 손실
        self.hp = hp_after
        # 연료
        speed = torch.linalg.norm(own[:, 6:9], dim=1)
        burn = R.FUEL_BURN_PER_SEC * (speed / R.FUEL_REF_SPEED)
        self.fuel = (self.fuel - burn * self.dt).clamp_min(0.0)
        # pqr (SO3 log)
        att = own[:, 3:6]
        Rb2n_prev = ned_to_body(self.prev_att).transpose(1, 2)
        Rb2n_curr = ned_to_body(att).transpose(1, 2)
        r_delta = torch.bmm(Rb2n_prev.transpose(1, 2), Rb2n_curr)
        pqr_new = log_so3(r_delta) / max(self.dt, 1e-8)
        self.pqr = torch.where(self.prev_valid.unsqueeze(1), pqr_new, torch.zeros_like(pqr_new))
        # 선가속도: NED 속도의 step 차분. prev_valid 를 pqr 과 공유하므로 반드시 아래
        # prev_valid 갱신 **전에** 계산해야 첫 step 이 0 으로 나온다.
        vel_ned = _mv(Rb2n_curr, own[:, 6:9])
        accel_new = (vel_ned - self.prev_vel_ned) / max(self.dt, 1e-8)
        self.accel = torch.where(self.prev_valid.unsqueeze(1), accel_new,
                                 torch.zeros_like(accel_new))
        self.prev_vel_ned = vel_ned.clone()
        self.prev_att = att.clone()
        self.prev_valid = torch.ones_like(self.prev_valid)
        # last dmg (rate)
        self.last_dmg_dealt = rate_deals
        self.last_dmg_taken = rate_taken
        # 시간
        self.t_sec = self.t_sec + self.dt

    # ── 관측 (nac, OBS_SIZE) ──────────────────────────────────────────────────
    def build_obs(self, states9):
        own = states9
        tgt = states9[self.partner]
        t_ac = self.t_sec[self.env_of]

        own_rpy = own[:, 3:6]; tgt_rpy = tgt[:, 3:6]
        Rnb_o = ned_to_body(own_rpy); Rb2n_o = Rnb_o.transpose(1, 2)
        Rnb_t = ned_to_body(tgt_rpy); Rb2n_t = Rnb_t.transpose(1, 2)
        own_vb = own[:, 6:9]; tgt_vb = tgt[:, 6:9]
        own_vn = _mv(Rb2n_o, own_vb)
        tgt_vn = _mv(Rb2n_t, tgt_vb)
        rel_vn = tgt_vn - own_vn
        own_spd = torch.linalg.norm(own_vb, dim=1)
        tgt_spd = torch.linalg.norm(tgt_vb, dim=1)
        own_alt = -own[:, 2]; tgt_alt = -tgt[:, 2]

        delta = tgt[:, :3] - own[:, :3]
        dist = torch.linalg.norm(delta, dim=1)
        los_u = torch.where(dist.unsqueeze(1) > 1e-6, delta / dist.clamp_min(1e-300).unsqueeze(1),
                            torch.zeros_like(delta))
        closure = torch.where(dist > 1e-6, ((own_vn - tgt_vn) * los_u).sum(1),
                              torch.zeros_like(dist))

        ata = ata_deg(own, tgt)                    # 0..180
        enemy_ata = ata[self.partner]
        aa = aspect_deg(own, tgt)
        az, el = los_az_el_deg(own, tgt)

        # ── AoA / sideslip ──
        u, v, w = own_vb[:, 0], own_vb[:, 1], own_vb[:, 2]
        slow = own_spd < 1.0
        aoa = torch.where(slow, torch.zeros_like(u), torch.atan2(w, u) * R2D)
        sslip = torch.where(slow, torch.zeros_like(u),
                            torch.atan2(v, torch.sqrt(u * u + w * w)) * R2D)
        vspeed = -own_vn[:, 2]

        e_own = own_alt + own_spd ** 2 / (2.0 * _G)
        e_tgt = tgt_alt + tgt_spd ** 2 / (2.0 * _G)
        e_adv = e_own - e_tgt

        s_ata, c_ata = _sincos_deg(ata)
        s_aa, c_aa = _sincos_deg(aa)
        s_az, c_az = _sincos_deg(az)
        s_el, c_el = _sincos_deg(el)

        aim_sharp = 2.0 * torch.exp(-((ata / 3.0) ** 2)) - 1.0
        cone, maxrng_ft = _active_envelope(t_ac)
        aim_margin = torch.tanh((cone - ata.abs()) / cone.clamp_min(1e-6))
        en_aim_sharp = 2.0 * torch.exp(-((enemy_ata / 3.0) ** 2)) - 1.0
        en_aim_margin = torch.tanh((cone - enemy_ata.abs()) / cone.clamp_min(1e-6))

        min_r_m = R.MIN_DAMAGE_RANGE_FT * _FT2M
        max_r_m = maxrng_ft * _FT2M
        span = (max_r_m - min_r_m).clamp_min(1e-6)
        rm_near = torch.tanh((dist - min_r_m) / span)
        rm_far = torch.tanh((max_r_m - dist) / span)

        or_s, or_c = _sincos_deg(own_rpy[:, 0])
        op_s, op_c = _sincos_deg(own_rpy[:, 1])
        oy_s, oy_c = _sincos_deg(own_rpy[:, 2])
        tr_s, tr_c = _sincos_deg(tgt_rpy[:, 0])
        tp_s, tp_c = _sincos_deg(tgt_rpy[:, 1])
        ty_s, ty_c = _sincos_deg(tgt_rpy[:, 2])
        ovb_s, ovb_c = bank_sincos(Rb2n_o, own_vn)
        tvb_s, tvb_c = bank_sincos(Rb2n_t, tgt_vn)

        dmg_dealt = self.last_dmg_dealt
        dmg_taken = self.last_dmg_taken
        pf_ata = (1.0 - ata.abs() / R.PURSUIT_ATA_SCALE_DEG).clamp_min(0.0)
        pf_rng = (1.0 - dist / R.PURSUIT_RANGE_M).clamp_min(0.0)
        pursuit = 2.0 * (pf_ata * pf_rng) - 1.0

        hp_o = self.hp; hp_t = self.hp[self.partner]
        fuel_o = self.fuel; fuel_t = self.fuel[self.partner]

        scalars = [
            normalize_t(own_spd, 0.0, R.MAX_SPEED),
            normalize_t(tgt_spd, 0.0, R.MAX_SPEED),
            torch.tanh(aoa / R.AOA_SCALE_DEG),
            torch.tanh(sslip / R.SIDESLIP_SCALE_DEG),
            torch.tanh((own_alt - R.MIN_ALTITUDE_M) / R.ALTITUDE_DANGER_SCALE_M),
            normalize_t(vspeed, -R.VERTICAL_SPEED_SCALE, R.VERTICAL_SPEED_SCALE),
            normalize_t(hp_o, 0.0, 1.0),
            normalize_t(hp_t, 0.0, 1.0),
            hp_o - hp_t,
            e_adv / (e_adv.abs() + R.ENERGY_ADVANTAGE_SCALE_M),
            normalize_t(dist, 0.0, R.MAX_RANGE_M),
            normalize_t(closure, -R.MAX_CLOSURE_SPEED, R.MAX_CLOSURE_SPEED),
            s_ata, c_ata, s_aa, c_aa, s_az, c_az, s_el, c_el,
            aim_sharp, aim_margin, en_aim_sharp, en_aim_margin,
            rm_near, rm_far,
            normalize_t(t_ac, 0.0, R.EPISODE_MAX_TIME_SEC),
            or_s, or_c, op_s, op_c, oy_s, oy_c,
            tr_s, tr_c, tp_s, tp_c, ty_s, ty_c,
            ovb_s, ovb_c, tvb_s, tvb_c,
            normalize_t(fuel_o, 0.0, 1.0),
            normalize_t(fuel_t, 0.0, 1.0),
            (2.0 * dmg_dealt - 1.0).clamp(-1.0, 1.0),
            (2.0 * dmg_taken - 1.0).clamp(-1.0, 1.0),
            pursuit,
            normalize_t(own_alt, 0.0, R.MAX_ALTITUDE_M),
            normalize_t(tgt_alt, 0.0, R.MAX_ALTITUDE_M),
        ]
        scal = torch.stack(scalars, 1)             # (nac,50)

        # ── frame-expressed 벡터 블록 (VEC_LAYOUT 순서) ──
        own_om_n = _mv(Rb2n_o, self.pqr)
        tgt_om_n = _mv(Rb2n_t, self.pqr[self.partner])
        frames = {
            "world": torch.eye(3, device=self.device, dtype=self.dtype).expand(self.nac, 3, 3),
            "mybody": Rnb_o, "oppbody": Rnb_t,
            "myvel": dir_frame(own_vn), "oppvel": dir_frame(tgt_vn),
            "los": dir_frame(delta),
        }
        vecs = {"gravity": torch.tensor([0.0, 0.0, 1.0], device=self.device,
                                        dtype=self.dtype).expand(self.nac, 3),
                "los": los_u, "own_vel": own_vn, "tgt_vel": tgt_vn, "rel_vel": rel_vn,
                "own_omega": own_om_n, "tgt_omega": tgt_om_n,
                "own_accel": self.accel, "tgt_accel": self.accel[self.partner]}
        vcols = []
        for key, kind, fr in VEC_LAYOUT:
            comp = _mv(frames[fr], vecs[key])       # (nac,3)
            if kind == "unit":
                vcols.append(comp)
            elif kind == "vel":
                vcols.append(normalize_t(comp, R.REL_VEL_MIN, R.REL_VEL_MAX))
            elif kind == "accel":
                vcols.append(normalize_t(comp, -R.ACCEL_SCALE_M_S2, R.ACCEL_SCALE_M_S2))
            else:  # omega
                vcols.append(torch.tanh(comp / R.PQR_SCALE_RAD_S))
        vecb = torch.cat(vcols, 1)                  # (nac,144)

        acth = self.act_hist.reshape(self.nac, -1)  # (nac,20)

        obs = torch.cat([scal, vecb, acth], 1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.to(torch.float32)

    @staticmethod
    def _altitude_log(states9):
        return ((-states9[:, 2]) / ALTITUDE_LOG_SCALE_M).clamp_min(ALTITUDE_LOG_FLOOR).log()

    @torch.no_grad()
    def initialize_reward_state(self, states9, env_mask=None, cfg=None):
        """Torch reference path의 reset tracker 초기화."""
        cfg = cfg if cfg is not None else RW.MY_REWARD_CONFIG
        own, tgt = states9, states9[self.partner]
        t_ac = self.t_sec[self.env_of]
        side = final_safe_geometry(own, tgt, t_ac, cfg)
        geometry = (side - side[self.partner]).clamp(-1.0, 1.0)
        alt_log = self._altitude_log(states9)
        if env_mask is None:
            self.prev_x.copy_(geometry); self.prev_x_valid.fill_(True)
            self.prev_alt_log.copy_(alt_log)
        else:
            ac = env_mask.repeat_interleave(2)
            self.prev_x[ac] = geometry[ac]
            self.prev_x_valid[ac] = True
            self.prev_alt_log[ac] = alt_log[ac]

    # ── 보상 (nac,) ──────────────────────────────────────────────────────────
    def compute_reward(self, states9, terminated_env, cfg=None, truncated_env=None,
                       reward_mode=0, alt_hunt_coef=5.0):
        """final-safe reward의 torch reference. fused CUDA 커널 검증용."""
        cfg = cfg if cfg is not None else RW.MY_REWARD_CONFIG
        mode = validate_reward_mode(reward_mode, alt_hunt_coef)
        own = states9
        tgt = states9[self.partner]
        hp_o = self.hp; hp_t = self.hp[self.partner]
        loss_o = self.hp_loss                        # own_damage
        loss_t = self.hp_loss[self.partner]          # target_damage
        own_w = float(cfg.get("own_damage_weight", 0.5))
        dmg_scale = float(cfg["damage_scale"])
        altitude_scale = float(cfg.get("altitude_settlement_scale", dmg_scale))
        r_damage = damage_reward(mode, loss_t, loss_o, dmg_scale, own_w)

        # final-safe geometry occupancy: S=50/20/20/10, G=S-own-S-enemy.
        t_ac = self.t_sec[self.env_of]
        side = final_safe_geometry(own, tgt, t_ac, cfg)
        geometry = (side - side[self.partner]).clamp(-1.0, 1.0)
        previous = torch.where(self.prev_x_valid, self.prev_x, geometry)
        shaping_mult = (float(cfg["shaping_reward_scale"])
                        / float(RW.MY_REWARD_CONFIG["shaping_reward_scale"]))
        r_shaping = (shaping_mult * float(cfg["geometry_episode_budget"])
                     / float(cfg["geometry_reference_duration_sec"])
                     * 0.5 * (previous + geometry) * self.dt)
        current_alt_log = self._altitude_log(states9)
        if mode == ALTITUDE_HUNT_REWARD:
            r_shaping = torch.where(
                self.prev_x_valid[self.partner],
                (self.prev_alt_log[self.partner] - current_alt_log[self.partner]) * float(alt_hunt_coef),
                torch.zeros_like(hp_o))
        self.prev_x.copy_(geometry); self.prev_x_valid.fill_(True)
        self.prev_alt_log.copy_(current_alt_log)

        # 종료 우선순위: own altitude → target altitude → HP, timeout은 HP 비교.
        term_ac = terminated_env.repeat_interleave(2)
        own_below = (-own[:, 2]) < R.MIN_ALTITUDE_M
        tgt_below = (-tgt[:, 2]) < R.MIN_ALTITUDE_M
        r_term = torch.zeros_like(hp_o)
        r_term = torch.where(term_ac & own_below,
                             -hp_o.clamp_min(0.0) * altitude_scale, r_term)
        r_term = torch.where(term_ac & (~own_below) & tgt_below,
                             hp_t.clamp_min(0.0) * altitude_scale, r_term)
        hp_only = term_ac & (~own_below) & (~tgt_below)
        r_term = r_term + torch.where(
            hp_only & (hp_t <= 0.0), torch.full_like(hp_o, float(cfg["win_reward"])),
            torch.zeros_like(hp_o))
        r_term = r_term + torch.where(
            hp_only & (hp_o <= 0.0), torch.full_like(hp_o, float(cfg["loss_reward"])),
            torch.zeros_like(hp_o))
        if cfg.get("altitude_terminal_mode", "remaining_hp") in ("result", "result_remaining_hp"):
            own_dead = own_below | (hp_o <= 0.0)
            tgt_dead = tgt_below | (hp_t <= 0.0)
            r_term = torch.where(term_ac & tgt_dead & ~own_dead,
                                 torch.full_like(hp_o, float(cfg["win_reward"])),
                                 torch.zeros_like(hp_o))
            r_term = torch.where(term_ac & own_dead & ~tgt_dead,
                                 torch.full_like(hp_o, float(cfg["loss_reward"])), r_term)
            r_term = torch.where(term_ac & tgt_below & ~own_dead,
                                 torch.full_like(hp_o, float(cfg.get("altitude_win_reward", cfg["win_reward"]))), r_term)
            r_term = torch.where(term_ac & own_below & ~tgt_dead,
                                 torch.full_like(hp_o, float(cfg.get("altitude_loss_reward", cfg["loss_reward"]))), r_term)
            if cfg["altitude_terminal_mode"] == "result_remaining_hp":
                r_term = r_term + torch.where(term_ac & tgt_below & ~own_dead,
                                              hp_t.clamp_min(0.0) * altitude_scale, torch.zeros_like(hp_o))
                r_term = r_term - torch.where(term_ac & own_below & ~tgt_dead,
                                              hp_o.clamp_min(0.0) * altitude_scale, torch.zeros_like(hp_o))
        if truncated_env is not None:
            trunc_ac = truncated_env.repeat_interleave(2)
            timeout = torch.where(
                hp_o > hp_t, torch.full_like(hp_o, float(cfg["timeout_win_reward"])),
                torch.where(hp_o < hp_t,
                            torch.full_like(hp_o, float(cfg["timeout_loss_reward"])),
                            torch.full_like(hp_o, float(cfg["timeout_draw_reward"]))))
            r_term = torch.where(trunc_ac, timeout, r_term)
        return r_damage + r_shaping + r_term
