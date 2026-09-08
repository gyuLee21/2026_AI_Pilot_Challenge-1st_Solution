# -*- coding: utf-8 -*-
"""Phase2 최종 보상: 결과 ±5, 순 피해 ±10, geometry occupancy 예산 ±5.

한 기체의 공격 geometry는 다음 단일 수식만 사용한다.

``S = 0.50 * broad + 0.20 * safe_lead + 0.20 * control_zone + 0.10 * fine``

Safe Lead는 먼 거리에서 0.3~1.5초 미래점을 향한 기수(40%)/비행경로(60%) 정렬을
사용하고, 가까워지거나 ATA가 작아지면 현재 표적 LOS에 대한 기수 정렬 100%로
smoothstep 전환한다. Control-zone K는 body-frame u/v/w를 NED로 회전한 뒤 거리·ATA·
정규화 closing rate를 함께 평가한다. Fine은 환경과 같은 phase1→phase2→phase3
우선순위를 사용한다.

최종 상대 점수 ``G=S_own-S_enemy``는 [-1, 1]이며 RL interval마다 사다리꼴 적분한다.
reset 시 최초 상태의 G를 tracker에 넣어 첫 interval도 초기→다음 상태 평균으로 계산한다.
감쇠 일정과 총 geometry 예산은 유지한다. 저고도 상태 자체의 shaping은 없다.

``shaping_reward_scale``은 기존 CLI/MPC 호환 수치 0.0001을 기준 배율 1로 정규화하며,
학습 스케줄에서 1/.6/.32/.12/0 비율로 감소한다.

고도이탈 종료는 추락한 기체의 남은 HP × damage_scale을 잃거나 얻는다.
기존 고도 potential과 고정 -15/+5는 합산하지 않는다. 다른 HP/timeout 종료 항은 유지한다.
추락 유도 exploiter(reward_mode=1)는 geometry 대신 C*log(상대 이전고도/현재고도)를
받는다. main 및 기본 exploiter는 reward_mode=0이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dogfight.sim.state_schema import StateIndex
from cuda_fdm.reward_modes import (ALTITUDE_HUNT_REWARD, log_altitude,
                                   validate_reward_mode, damage_reward)


_FT_TO_M = 0.3048

MY_REWARD_CONFIG = {
    # Phase2 결과/피해 계약. geometry는 200초 동안 최대 우위를 계속 유지했을 때
    # episode 합계가 +5(최대 열세는 -5)가 되는 점유형 보상이다.
    "win_reward": 5.0,
    "loss_reward": -5.0,
    # Legacy CPU/MPC full-HP fallback keys only. compute_reward and CUDA calculate
    # the actual altitude settlement dynamically from remaining HP, not these constants.
    "ownship_alt_reward": -10.0,
    "target_alt_reward": 10.0,
    "altitude_terminal_mode": "remaining_hp",
    "reward_mode": 0,
    "alt_hunt_coef": 5.0,
    "timeout_win_reward": 5.0,
    "timeout_loss_reward": -5.0,
    "timeout_draw_reward": -4.0,
    "damage_scale": 10.0,   # (상대 HP감소 - 내 HP감소*own_damage_weight) * 이 값, 양측 생존 중 매 step
    "own_damage_weight": 1.0,
    "geometry_episode_budget": 5.0,
    "geometry_reference_duration_sec": 200.0,
    # 비교 분기 없이 이 단일 점수/적분만 최종 학습에 사용한다.
    "geometry_score_version": "final_safe",
    "geometry_integration": "trapezoid",
    "broad_ata_full_control_angle_deg": 30.0,
    "weapon_min_range_ft": 500.0,
    "weapon_max_range_ft": 4000.0,
    "weapon_full_alignment_deg": 3.0,
    "weapon_transition_angle_deg": 8.0,
    "damage_full_alignment_deg": 1.0,
    "tier2_start_sec": 100.0,
    "tier3_start_sec": 150.0,
    "tier1_max_range_ft": 3000.0,
    "tier2_max_range_ft": 3500.0,
    "tier3_max_range_ft": 4000.0,
    # Safe Lead: 먼 거리 미래점과 사격 직전 현재 LOS 사이의 부드러운 handoff.
    "lead_prediction_min_sec": 0.30,
    "lead_prediction_max_sec": 1.50,
    "lead_speed_floor_mps": 30.0,
    "lead_nose_weight": 0.40,
    "lead_path_weight": 0.60,
    "lead_direct_range_ft": 1000.0,
    "lead_full_range_ft": 2500.0,
    "lead_direct_ata_deg": 3.0,
    "lead_full_ata_deg": 8.0,
    "alignment_power": 3.0,
    # Control-zone K. 거리 단위는 ft, 속도 단위는 NED m/s, closure는 무차원이다.
    "control_zone_near_zero_ft": 600.0,
    "control_zone_near_full_ft": 900.0,
    "control_zone_far_full_ft": 3200.0,
    "control_zone_far_zero_ft": 4300.0,
    "control_zone_far_closure_start_ft": 2000.0,
    "control_zone_far_closure_full_ft": 4500.0,
    "control_zone_close_opening_full_ft": 650.0,
    "control_zone_close_opening_zero_ft": 900.0,
    "control_zone_far_closure_ratio": 0.20,
    "control_zone_close_opening_ratio": -0.12,
    "control_zone_closure_sigma": 0.15,
    "control_zone_speed_floor_mps": 100.0,
    # 기존 CLI/MPC 수치 규약. compute_reward 내부에서 0.0001을 배율 1로 환산한다.
    "shaping_reward_scale": 0.0001,
}
# 주의: 아래 compute_reward 는 이 dict 의 키를 **직접 인덱싱**한다(.get 폴백 없음).
# 계수를 바꾸려면 반드시 이 dict(또는 train.py 의 --*-reward-scale 오버라이드)를 고칠 것.
# 예전에는 .get(key, 폴백) 형태라 폴백만 고치면 아무 효과가 없는 함정이 있었다.

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# 공식 min_altitude(1000ft=304.8m MSL)에서 정확히 발생한다.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"

# dense shaping 직전-step 상태 (모듈 싱글톤). 에피소드 경계는 SIM_TIME 으로 감지.
_prev_sim_time: float | None = None
_prev_target_log_altitude: float | None = None
_prev_geometry_advantage: float | None = None


def _body_to_ned_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """JSBSim body-frame 벡터를 NED로 회전하는 3-2-1 DCM."""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ], dtype=np.float64)


def _velocity_ned(state) -> np.ndarray:
    """State의 body u/v/w(m/s)를 inertial NED(m/s)로 변환한다."""
    value = np.asarray(state, dtype=np.float64)
    rotation = _body_to_ned_matrix(
        value[StateIndex.ROLL], value[StateIndex.PITCH], value[StateIndex.YAW])
    return rotation @ value[6:9]


def _unit_vector(vector, fallback=None) -> np.ndarray:
    """유한한 3D 단위벡터. 영벡터면 명시한 fallback을 사용한다."""
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm >= 1.0e-9:
        return value / norm
    if fallback is not None:
        fallback_value = np.asarray(fallback, dtype=np.float64)
        fallback_norm = float(np.linalg.norm(fallback_value))
        if np.isfinite(fallback_norm) and fallback_norm >= 1.0e-9:
            return fallback_value / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _alignment_quality(vector, direction, power: float = 3.0) -> float:
    """Q(a,b)=((1+dot(a_hat,b_hat))/2)^power를 [0,1]로 반환한다."""
    a = _unit_vector(vector)
    b = _unit_vector(direction)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    score = ((1.0 + cosine) * 0.5) ** float(power)
    if not np.isfinite(score):
        raise FloatingPointError(f"non-finite alignment quality: {score}")
    return float(np.clip(score, 0.0, 1.0))


def _safe_lead_blend_weight(distance_ft: float, ata_deg: float, config: dict) -> float:
    """먼 거리/큰 ATA에서 1, 근거리/작은 ATA에서 0인 Lead 비율 w."""
    weight = (
        _smoothstep(
            distance_ft,
            float(config["lead_direct_range_ft"]),
            float(config["lead_full_range_ft"]),
        )
        * _smoothstep(
            abs(float(ata_deg)),
            float(config["lead_direct_ata_deg"]),
            float(config["lead_full_ata_deg"]),
        )
    )
    return float(np.clip(weight, 0.0, 1.0))


def _safe_lead_from_vectors(origin, destination, forward, own_velocity,
                            destination_velocity, distance_ft: float,
                            ata_deg: float, config: dict) -> tuple[float, dict]:
    """미리 NED로 변환한 벡터로 Safe Lead 점수와 진단값을 계산한다."""
    origin = np.asarray(origin, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    forward = _unit_vector(forward)
    own_velocity = np.asarray(own_velocity, dtype=np.float64)
    destination_velocity = np.asarray(destination_velocity, dtype=np.float64)
    line_of_sight = _unit_vector(destination - origin, fallback=forward)
    path_direction = _unit_vector(own_velocity, fallback=forward)

    distance_m = max(0.0, float(distance_ft)) * _FT_TO_M
    own_speed = max(
        float(np.linalg.norm(own_velocity)),
        float(config["lead_speed_floor_mps"]),
    )
    tau = float(np.clip(
        distance_m / own_speed,
        float(config["lead_prediction_min_sec"]),
        float(config["lead_prediction_max_sec"]),
    ))
    lead_direction = _unit_vector(
        destination + destination_velocity * tau - origin,
        fallback=line_of_sight,
    )
    power = float(config["alignment_power"])
    nose_to_lead = _alignment_quality(forward, lead_direction, power)
    path_to_lead = _alignment_quality(path_direction, lead_direction, power)
    nose_weight = float(config["lead_nose_weight"])
    path_weight = float(config["lead_path_weight"])
    weight_sum = nose_weight + path_weight
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Safe Lead nose/path weights must have a positive finite sum")
    intercept = (nose_weight * nose_to_lead + path_weight * path_to_lead) / weight_sum
    direct = _alignment_quality(forward, line_of_sight, power)
    lead_weight = _safe_lead_blend_weight(distance_ft, ata_deg, config)
    score = lead_weight * intercept + (1.0 - lead_weight) * direct
    details = {
        "lead_intercept": float(np.clip(intercept, 0.0, 1.0)),
        "lead_direct": float(np.clip(direct, 0.0, 1.0)),
        "lead_weight": lead_weight,
        "lead_tau_sec": tau,
    }
    if not np.isfinite(score) or not all(np.isfinite(v) for v in details.values()):
        raise FloatingPointError(f"non-finite Safe Lead score: {score}, {details}")
    return float(np.clip(score, 0.0, 1.0)), details


def _safe_lead_score(ownship_state, target_state, distance_ft: float,
                     ata_deg: float, config: dict) -> tuple[float, dict]:
    """상태의 body 속도를 NED로 바꿔 최종 Safe Lead 점수를 계산한다."""
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    own_rotation = _body_to_ned_matrix(
        own[StateIndex.ROLL], own[StateIndex.PITCH], own[StateIndex.YAW])
    return _safe_lead_from_vectors(
        own[:3], target[:3], own_rotation[:, 0],
        own_rotation @ own[6:9], _velocity_ned(target),
        distance_ft, ata_deg, config,
    )


def _broad_ata_control(ata_deg: float, full_angle_deg: float = 30.0) -> float:
    """Phase2 broad ATA control: 0~30도는 1, 이후 180도까지 선형으로 0."""
    ata = float(np.clip(abs(ata_deg), 0.0, 180.0))
    if ata <= full_angle_deg:
        return 1.0
    return float(np.clip((180.0 - ata) / (180.0 - full_angle_deg), 0.0, 1.0))


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    """edge0에서 0, edge1에서 1인 cubic smoothstep."""
    if not edge1 > edge0:
        raise ValueError(f"smoothstep edge1 must exceed edge0: {edge0}, {edge1}")
    z = float(np.clip((float(value) - edge0) / (edge1 - edge0), 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


def _control_zone_score(ownship_state, target_state, distance_ft: float,
                        ata_deg: float, config: dict) -> tuple[float, dict]:
    """거리·ATA·closing rate를 함께 만족하는 지속 가능한 공격 품질 K를 반환한다.

    ``closing_ratio > 0``은 거리가 줄어드는 상태다. 두 기체의 body u/v/w는 먼저 같은
    NED frame으로 회전한다. 완전 겹친 위치나 비정상 저속에서도 NaN/Inf가 생기지 않는다.
    """
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    relative_position = target[:3] - own[:3]
    relative_norm = float(np.linalg.norm(relative_position))

    angle_gate = 1.0 - _smoothstep(
        abs(float(ata_deg)),
        float(config["weapon_full_alignment_deg"]),
        float(config["weapon_transition_angle_deg"]),
    )
    range_gate = (
        _smoothstep(distance_ft,
                    float(config["control_zone_near_zero_ft"]),
                    float(config["control_zone_near_full_ft"]))
        * (1.0 - _smoothstep(
            distance_ft,
            float(config["control_zone_far_full_ft"]),
            float(config["control_zone_far_zero_ft"])))
    )

    desired_closing = (
        float(config["control_zone_far_closure_ratio"])
        * _smoothstep(
            distance_ft,
            float(config["control_zone_far_closure_start_ft"]),
            float(config["control_zone_far_closure_full_ft"]))
        + float(config["control_zone_close_opening_ratio"])
        * (1.0 - _smoothstep(
            distance_ft,
            float(config["control_zone_close_opening_full_ft"]),
            float(config["control_zone_close_opening_zero_ft"])))
    )

    closing_ratio = 0.0
    if relative_norm >= 1.0e-9:
        los = relative_position / relative_norm
        own_velocity = _velocity_ned(own)
        target_velocity = _velocity_ned(target)
        closing_speed = -float(np.dot(los, target_velocity - own_velocity))
        mean_speed = 0.5 * (
            float(np.linalg.norm(own_velocity)) + float(np.linalg.norm(target_velocity)))
        denominator = max(mean_speed, float(config["control_zone_speed_floor_mps"]))
        closing_ratio = closing_speed / denominator

    sigma = max(float(config["control_zone_closure_sigma"]), 1.0e-6)
    closure_quality = float(np.exp(
        -0.5 * ((closing_ratio - desired_closing) / sigma) ** 2))
    score = float(np.clip(angle_gate * range_gate * closure_quality, 0.0, 1.0))
    details = {
        "angle_gate": float(np.clip(angle_gate, 0.0, 1.0)),
        "range_gate": float(np.clip(range_gate, 0.0, 1.0)),
        "closure_quality": float(np.clip(closure_quality, 0.0, 1.0)),
        "closing_ratio": float(closing_ratio),
        "desired_closing_ratio": float(desired_closing),
    }
    if not np.isfinite(score) or not all(np.isfinite(v) for v in details.values()):
        raise FloatingPointError(f"non-finite control-zone score: {score}, {details}")
    return score, details


def _range_effectiveness(distance_ft: float, config: dict) -> float:
    """실제 damage 식과 같은 거리 효율: 500ft=1, 3000ft=0, 범위 밖=0."""
    lo = float(config["weapon_min_range_ft"])
    hi = float(config["weapon_max_range_ft"])
    if not (lo <= distance_ft <= hi):
        return 0.0
    return float(np.clip((hi - distance_ft) / max(hi - lo, 1.0e-6), 0.0, 1.0))


def _alignment_gate(distance_ft: float, ata_deg: float, full: float, edge: float,
                    config: dict) -> float:
    """실제 WEZ 안에서 거리 효율×각도 정렬도를 [0,1]로 반환한다."""
    if not (float(config["weapon_min_range_ft"]) <= distance_ft
            <= float(config["weapon_max_range_ft"])):
        return 0.0
    ata = abs(float(ata_deg))
    if ata <= full:
        angular = 1.0
    elif ata >= edge:
        angular = 0.0
    else:
        t = (ata - full) / max(edge - full, 1.0e-6)
        angular = float(1.0 - t * t * (3.0 - 2.0 * t))
    return _range_effectiveness(distance_ft, config) * angular


def _actual_damage_score(distance_ft: float, ata_deg: float, sim_time: float,
                         config: dict) -> float:
    """TierGatedDogFightEnv.damage_rate와 동일한 [0,1] damage rate."""
    lo = float(config["weapon_min_range_ft"])
    r, a, t = float(distance_ft), abs(float(ata_deg)), float(sim_time)
    r1, r2, r3 = (float(config["tier1_max_range_ft"]),
                  float(config["tier2_max_range_ft"]),
                  float(config["tier3_max_range_ft"]))
    # 환경처럼 낮은 tier를 먼저 검사한다. 경계에서 rate=0인 것도 그대로 일치시킨다.
    if lo <= r <= r1 and a < 1.0:
        return float((r1 - r) / max(r1 - lo, 1.0e-6))
    if t >= float(config["tier2_start_sec"]) and lo <= r <= r2 and a < 2.0:
        return float(0.3 * (r2 - r) / max(r2 - lo, 1.0e-6))
    if t >= float(config["tier3_start_sec"]) and lo <= r <= r3 and a < 3.0:
        return float(0.1 * (r3 - r) / max(r3 - lo, 1.0e-6))
    return 0.0


def _geometry_side_terms(ownship_state, target_state, distance_ft: float,
                         ata_deg: float, sim_time: float, config: dict) -> dict:
    """한 기체가 다른 기체를 공격하는 관점의 bounded geometry 구성 요소."""
    full = float(config["broad_ata_full_control_angle_deg"])
    coarse_full = float(config["weapon_full_alignment_deg"])
    coarse_edge = float(config["weapon_transition_angle_deg"])
    control_zone, control_details = _control_zone_score(
        ownship_state, target_state, distance_ft, ata_deg, config)
    safe_lead, lead_details = _safe_lead_score(
        ownship_state, target_state, distance_ft, ata_deg, config)
    terms = {
        "broad": _broad_ata_control(ata_deg, full),
        # 진단 키 이름은 기존 로그 호환을 위해 lead지만 값은 최종 L_safe다.
        "lead": safe_lead,
        "coarse": _alignment_gate(
            distance_ft, ata_deg, coarse_full, coarse_edge, config),
        # 실제 환경처럼 phase1 -> phase2 -> phase3 priority를 사용한다.
        "fine": _actual_damage_score(distance_ft, ata_deg, sim_time, config),
        "control_zone": control_zone,
        **lead_details,
        **control_details,
    }
    bounded = ("broad", "lead", "coarse", "fine", "control_zone",
               "lead_intercept", "lead_direct", "lead_weight",
               "angle_gate", "range_gate", "closure_quality")
    for key in bounded:
        value = float(terms[key])
        if not np.isfinite(value) or not -1.0e-12 <= value <= 1.0 + 1.0e-12:
            raise FloatingPointError(f"geometry component {key} out of [0,1]: {value}")
        terms[key] = float(np.clip(value, 0.0, 1.0))
    return terms


def _validate_geometry_terms(terms: dict) -> dict:
    bounded = ("broad", "lead", "coarse", "fine", "control_zone",
               "lead_intercept", "lead_direct", "lead_weight",
               "angle_gate", "range_gate", "closure_quality")
    for key in bounded:
        value = float(terms[key])
        if not np.isfinite(value) or not -1.0e-12 <= value <= 1.0 + 1.0e-12:
            raise FloatingPointError(f"geometry component {key} out of [0,1]: {value}")
        terms[key] = float(np.clip(value, 0.0, 1.0))
    return terms


def _geometry_pair_terms(ownship_state, target_state, distance_ft: float,
                         own_ata_deg: float, enemy_ata_deg: float,
                         sim_time: float, config: dict) -> tuple[dict, dict]:
    """양측 공통 DCM/속도/closure를 한 번만 계산하는 rollout hot path."""
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    own_rotation = _body_to_ned_matrix(
        own[StateIndex.ROLL], own[StateIndex.PITCH], own[StateIndex.YAW])
    target_rotation = _body_to_ned_matrix(
        target[StateIndex.ROLL], target[StateIndex.PITCH], target[StateIndex.YAW])
    own_velocity = own_rotation @ own[6:9]
    target_velocity = target_rotation @ target[6:9]
    relative_position = target[:3] - own[:3]
    relative_norm = float(np.linalg.norm(relative_position))

    own_lead, own_lead_details = _safe_lead_from_vectors(
        own[:3], target[:3], own_rotation[:, 0], own_velocity, target_velocity,
        distance_ft, own_ata_deg, config)
    enemy_lead, enemy_lead_details = _safe_lead_from_vectors(
        target[:3], own[:3], target_rotation[:, 0], target_velocity, own_velocity,
        distance_ft, enemy_ata_deg, config)

    closing_ratio = 0.0
    if relative_norm >= 1.0e-9:
        los = relative_position / relative_norm
        closing_speed = -float(np.dot(los, target_velocity - own_velocity))
        mean_speed = 0.5 * (
            float(np.linalg.norm(own_velocity)) + float(np.linalg.norm(target_velocity)))
        closing_ratio = closing_speed / max(
            mean_speed, float(config["control_zone_speed_floor_mps"]))
    desired_closing = (
        float(config["control_zone_far_closure_ratio"])
        * _smoothstep(
            distance_ft,
            float(config["control_zone_far_closure_start_ft"]),
            float(config["control_zone_far_closure_full_ft"]))
        + float(config["control_zone_close_opening_ratio"])
        * (1.0 - _smoothstep(
            distance_ft,
            float(config["control_zone_close_opening_full_ft"]),
            float(config["control_zone_close_opening_zero_ft"])))
    )
    sigma = max(float(config["control_zone_closure_sigma"]), 1.0e-6)
    closure_quality = float(np.exp(
        -0.5 * ((closing_ratio - desired_closing) / sigma) ** 2))
    range_gate = (
        _smoothstep(distance_ft,
                    float(config["control_zone_near_zero_ft"]),
                    float(config["control_zone_near_full_ft"]))
        * (1.0 - _smoothstep(
            distance_ft,
            float(config["control_zone_far_full_ft"]),
            float(config["control_zone_far_zero_ft"])))
    )

    full = float(config["broad_ata_full_control_angle_deg"])
    coarse_full = float(config["weapon_full_alignment_deg"])
    coarse_edge = float(config["weapon_transition_angle_deg"])

    def make_terms(ata_deg, lead, lead_details):
        angle_gate = 1.0 - _smoothstep(abs(float(ata_deg)), coarse_full, coarse_edge)
        return _validate_geometry_terms({
            "broad": _broad_ata_control(ata_deg, full),
            "lead": lead,
            **lead_details,
            "coarse": _alignment_gate(
                distance_ft, ata_deg, coarse_full, coarse_edge, config),
            "fine": _actual_damage_score(distance_ft, ata_deg, sim_time, config),
            "control_zone": float(np.clip(
                angle_gate * range_gate * closure_quality, 0.0, 1.0)),
            "angle_gate": angle_gate,
            "range_gate": range_gate,
            "closure_quality": closure_quality,
            "closing_ratio": closing_ratio,
            "desired_closing_ratio": desired_closing,
        })

    return (make_terms(own_ata_deg, own_lead, own_lead_details),
            make_terms(enemy_ata_deg, enemy_lead, enemy_lead_details))


def _geometry_side_score(terms: dict, version: str, config: dict) -> float:
    """최종 50/20/20/10 단일 수식으로 side score를 만든다."""
    version = str(version).lower()
    if version != "final_safe":
        raise ValueError(
            f"unknown geometry_score_version={version!r}; expected final_safe")
    score = (0.50 * terms["broad"] + 0.20 * terms["lead"]
             + 0.20 * terms["control_zone"] + 0.10 * terms["fine"])
    if not np.isfinite(score) or not -1.0e-12 <= score <= 1.0 + 1.0e-12:
        raise FloatingPointError(f"geometry side score out of [0,1]: {score}")
    return float(np.clip(score, 0.0, 1.0))


def _geometry_advantage(ownship_state, target_state, distance_ft: float,
                        own_ata_deg: float, enemy_ata_deg: float, sim_time: float,
                        config: dict, *, return_details: bool = False):
    """대칭 relative geometry ``S_own-S_enemy``와 선택적 진단값을 반환한다."""
    version = str(config["geometry_score_version"]).lower()
    own_terms, enemy_terms = _geometry_pair_terms(
        ownship_state, target_state, distance_ft, own_ata_deg, enemy_ata_deg,
        sim_time, config)
    own_score = _geometry_side_score(own_terms, version, config)
    enemy_score = _geometry_side_score(enemy_terms, version, config)
    unbounded = own_score - enemy_score
    if not -1.0 - 1.0e-12 <= unbounded <= 1.0 + 1.0e-12:
        raise FloatingPointError(f"geometry advantage out of [-1,1]: {unbounded}")
    advantage = float(np.clip(unbounded, -1.0, 1.0))
    if not return_details:
        return advantage
    return advantage, {
        "version": version,
        "own_score": own_score,
        "enemy_score": enemy_score,
        "own": own_terms,
        "enemy": enemy_terms,
    }


def _integrated_geometry_value(previous: float, current: float, dt: float,
                               mode: str) -> float:
    """한 RL interval의 occupancy integral. 60Hz 적분을 흉내내지 않는다."""
    mode = str(mode).lower()
    if mode == "endpoint":
        interval_score = float(current)
    elif mode == "trapezoid":
        interval_score = 0.5 * (float(previous) + float(current))
    else:
        raise ValueError(
            f"unknown geometry_integration={mode!r}; expected endpoint/trapezoid")
    return max(0.0, float(dt)) * interval_score


def initialize_reward_episode(ownship_state, target_state, geo_info,
                              reward_config: dict) -> None:
    """reset 직후 geometry 및 상대 log 고도 tracker를 최초 상태로 초기화한다."""
    global _prev_sim_time, _prev_target_log_altitude, _prev_geometry_advantage
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    _prev_sim_time = cur_sim_time
    _prev_target_log_altitude = log_altitude(float(target_state[StateIndex.ALT]))
    if float(reward_config["shaping_reward_scale"]) == 0.0:
        _prev_geometry_advantage = None
        return
    distance_ft = float(
        geo_info._get_distance(ownship_state, target_state)) / _FT_TO_M
    own_ata = abs(float(
        geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    enemy_ata = abs(float(
        geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
    _prev_geometry_advantage = _geometry_advantage(
        ownship_state, target_state, distance_ft, own_ata, enemy_ata,
        cur_sim_time, reward_config)


def reset_distance_tracker() -> None:
    """에피소드 시작 시 reward potential tracker를 초기화한다(기존 이름 호환용)."""
    global _prev_sim_time, _prev_target_log_altitude, _prev_geometry_advantage
    _prev_sim_time = None
    _prev_target_log_altitude = None
    _prev_geometry_advantage = None


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """순피해 + geometry(또는 상대고도 사냥) + 잔여 HP 고도이탈 정산."""
    global _prev_sim_time, _prev_target_log_altitude, _prev_geometry_advantage
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])

    # [보조] 순 HP 피해(terminal step 포함). 전체 HP 교환의 총량은 최대 ±10이다.
    r_damage = 0.0
    own_w = float(reward_config.get("own_damage_weight", 1.0))
    r_damage = damage_reward(reward_config.get("reward_mode", 0),
        float(target_damage), float(ownship_damage),
        float(reward_config["damage_scale"]), own_w)

    # [보조] 대칭 geometry 점유 shaping. 거리는 WEZ gate/거리효율에만 쓰며, 특정 목표
    # 거리로 끌어당기는 reward는 없다. SIM_TIME으로 episode 경계를 감지한다.
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    new_episode = _prev_sim_time is None or cur_sim_time <= _prev_sim_time

    mode = validate_reward_mode(reward_config.get("reward_mode", 0),
                                float(reward_config.get("alt_hunt_coef", 5.0)))
    target_altitude_m = float(target_state[StateIndex.ALT])
    target_log_altitude = log_altitude(target_altitude_m)
    r_alt_hunt = 0.0
    if mode == ALTITUDE_HUNT_REWARD and not new_episode and _prev_target_log_altitude is not None:
        r_alt_hunt = float(reward_config.get("alt_hunt_coef", 5.0)) * (
            _prev_target_log_altitude - target_log_altitude)
    _prev_target_log_altitude = target_log_altitude
    r_safety = 0.0  # legacy logging column; own-altitude shaping is removed

    shaping_scale = float(reward_config["shaping_reward_scale"])
    shaping_multiplier = shaping_scale / float(MY_REWARD_CONFIG["shaping_reward_scale"])
    r_shaping = 0.0
    r_geometry = 0.0
    r_distance = 0.0
    geometry_metrics = {
        "geometry_broad_delta_time": 0.0,
        "geometry_lead_delta_time": 0.0,
        "geometry_coarse_delta_time": 0.0,
        "geometry_control_zone_delta_time": 0.0,
        "geometry_fine_delta_time": 0.0,
        "control_zone_time": 0.0,
        "control_zone_score_time": 0.0,
        "control_zone_angle_gate_time": 0.0,
        "control_zone_range_gate_time": 0.0,
        "control_zone_closure_quality_time": 0.0,
        "bad_close_time": 0.0,
        "high_k_zero_damage_time": 0.0,
        "outer_high_k_time": 0.0,
        "geometry_metric_time": 0.0,
        "closing_ratio_time": 0.0,
    }
    if shaping_scale != 0.0 and mode != ALTITUDE_HUNT_REWARD:
        # _get_distance 는 meter → ft 로 환산. A1/A2 는 3D ATA(proj=False, 0~180).
        dist_ft = float(geo_info._get_distance(ownship_state, target_state)) / _FT_TO_M
        a1 = abs(float(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
        a2 = abs(float(
            geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
        geometry, geometry_details = _geometry_advantage(
            ownship_state, target_state, dist_ft, a1, a2, cur_sim_time,
            reward_config, return_details=True)
        if not new_episode:
            dt = max(0.0, cur_sim_time - float(_prev_sim_time))
            previous_geometry = (
                geometry if _prev_geometry_advantage is None
                else float(_prev_geometry_advantage))
            geometry_integral = _integrated_geometry_value(
                previous_geometry, geometry, dt,
                str(reward_config["geometry_integration"]))
            raw_geometry = (float(reward_config["geometry_episode_budget"])
                            / float(reward_config["geometry_reference_duration_sec"])
                            * geometry_integral)
            r_geometry = shaping_multiplier * raw_geometry
            r_shaping = r_geometry
            own_terms, enemy_terms = geometry_details["own"], geometry_details["enemy"]
            for name in ("broad", "lead", "coarse", "control_zone", "fine"):
                geometry_metrics[f"geometry_{name}_delta_time"] = dt * (
                    float(own_terms[name]) - float(enemy_terms[name]))
            own_k = float(own_terms["control_zone"])
            own_angle_gate = float(own_terms["angle_gate"])
            own_closing_ratio = float(own_terms["closing_ratio"])
            geometry_metrics["geometry_metric_time"] = dt
            geometry_metrics["closing_ratio_time"] = dt * own_closing_ratio
            geometry_metrics["control_zone_time"] = dt * float(own_k > 0.6)
            geometry_metrics["control_zone_score_time"] = dt * own_k
            geometry_metrics["control_zone_angle_gate_time"] = (
                dt * own_angle_gate)
            geometry_metrics["control_zone_range_gate_time"] = (
                dt * float(own_terms["range_gate"]))
            geometry_metrics["control_zone_closure_quality_time"] = (
                dt * float(own_terms["closure_quality"]))
            geometry_metrics["bad_close_time"] = dt * float(
                own_angle_gate > 0.5 and dist_ft < 900.0 and own_closing_ratio > 0.15)
            geometry_metrics["high_k_zero_damage_time"] = dt * float(
                own_k > 0.7 and float(target_damage) <= 0.0)
            geometry_metrics["outer_high_k_time"] = dt * float(
                own_k > 0.7 and dist_ft > 3000.0)
        _prev_geometry_advantage = geometry
    elif new_episode:
        _prev_geometry_advantage = None
    _prev_sim_time = cur_sim_time

    # Optional independent altitude scale preserves remaining-HP settlement
    # when a run deliberately retunes the dense damage coefficient.
    altitude_scale = float(reward_config.get("altitude_settlement_scale", reward_config["damage_scale"]))
    # [종료] 고도이탈은 남은 HP만 정산한다. 이번 step의 실제 피해는 위에서 이미 지급됐다.
    r_terminal = 0.0
    if terminated:
        if reward_config.get("altitude_terminal_mode", "remaining_hp") in ("result", "result_remaining_hp"):
            own_dead = own_hp <= 0.0 or float(ownship_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M
            tgt_dead = tgt_hp <= 0.0 or float(target_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M
            if tgt_dead and not own_dead:
                r_terminal = float(reward_config.get("altitude_win_reward", reward_config["win_reward"]) if
                                   float(target_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M else reward_config["win_reward"])
            elif own_dead and not tgt_dead:
                r_terminal = float(reward_config.get("altitude_loss_reward", reward_config["loss_reward"]) if
                                   float(ownship_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M else reward_config["loss_reward"])
            if reward_config["altitude_terminal_mode"] == "result_remaining_hp":
                if tgt_dead and not own_dead and float(target_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M:
                    r_terminal += max(0.0, tgt_hp) * altitude_scale
                elif own_dead and not tgt_dead and float(ownship_state[StateIndex.ALT]) < 1000.0 * _FT_TO_M:
                    r_terminal -= max(0.0, own_hp) * altitude_scale
        elif end_condition == _OWNSHIP_ALT_END:
            r_terminal = -max(0.0, own_hp) * altitude_scale
        elif end_condition == _TARGET_ALT_END:
            r_terminal = max(0.0, tgt_hp) * altitude_scale
        else:
            if tgt_hp <= 0.0:
                r_terminal += float(reward_config["win_reward"])
            if own_hp <= 0.0:
                r_terminal += float(reward_config["loss_reward"])
    elif truncated:
        if own_hp > tgt_hp:
            r_terminal = float(reward_config["timeout_win_reward"])
        elif own_hp < tgt_hp:
            r_terminal = float(reward_config["timeout_loss_reward"])
        else:
            r_terminal = float(reward_config["timeout_draw_reward"])

    total = r_damage + r_shaping + r_alt_hunt + r_terminal
    components = {"damage": r_damage, "geometry": r_geometry,
                  "distance": r_distance, "shaping": r_shaping,
                  "safety": r_safety, "altitude_hunt": r_alt_hunt, "terminal": r_terminal,
                  **geometry_metrics}
    return float(total), components


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
    "initialize_reward_episode",
    "reset_distance_tracker",
]
