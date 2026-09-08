"""Reward-mode contract shared by CPU reference, CUDA wrapper and exploiter.

Main/standard: final-safe geometry + damage + task terminal rewards.
Altitude hunter: opponent log-altitude progress replaces geometry, not damage
or terminal rewards. Attack/defense change ONLY the damage term. The scheduled
hunter occupies one in three slots; the remaining slots rotate all three styles.
(2026-09-04: the hunter-cycle sentence above describes the LEGACY archive-None
fallback only. In league mode the profile comes from the bandit staged fields
plus the ALTITUDE_SENTINEL_* forced hunter; scheduled_exploiter_mode and
EXPLOITER_CYCLE are not consulted. Contract string kept for compatibility.)
Modes never change a policy's observations or remaining-HP altitude settlement.
"""
import math

STANDARD_REWARD = 0
ALTITUDE_HUNT_REWARD = 1
ATTACK_REWARD = 2
DEFENSE_REWARD = 3
REWARD_CONTRACT = "remaining_hp_altitude_exit_1000ft_msl_damage_styles_hunt_each3_v3"
# One hunter per THREE slots, balanced/attack/defense each twice per NINE.
# Iteration 500/2000/3500/... are hunters. The other styles never add work.
# (2026-09-04: legacy archive-None fallback only; league mode uses the bandit
# staged fields + sentinel override instead. Values kept for compatibility.)
EXPLOITER_CYCLE = (ALTITUDE_HUNT_REWARD, STANDARD_REWARD, ATTACK_REWARD,
                  ALTITUDE_HUNT_REWARD, DEFENSE_REWARD, STANDARD_REWARD,
                  ALTITUDE_HUNT_REWARD, ATTACK_REWARD, DEFENSE_REWARD)
ALTITUDE_LOG_SCALE_M = 304.8  # 1000ft; cancels in differences
ALTITUDE_LOG_FLOOR = 1e-4


def validate_reward_mode(mode, coefficient=5.0):
    if mode not in (STANDARD_REWARD, ALTITUDE_HUNT_REWARD, ATTACK_REWARD,
                    DEFENSE_REWARD):
        raise ValueError(f"unknown reward mode: {mode!r}")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("altitude-hunt coefficient must be finite and nonnegative")
    return int(mode)


def log_altitude(altitude_m):
    if not math.isfinite(altitude_m):
        raise FloatingPointError("non-finite altitude reward input")
    return math.log(max(altitude_m / ALTITUDE_LOG_SCALE_M, ALTITUDE_LOG_FLOOR))


def scheduled_exploiter_mode(iteration, milestone_period, alternate=True):
    """One hunt in three slots; other slots cycle balanced/attack/defense.

    Derived from main iteration, never a mutable counter (resume is stable).
    The legacy `alternate` opt-out still disables specialized rewards entirely.
    Unscheduled/manual calls use standard reward. No milestones are added.

    2026-09-04: LEGACY archive-None fallback only. League mode never calls
    this (profile comes from the bandit staged fields + sentinel override).
    """
    if not alternate or milestone_period <= 0 or iteration <= 0:
        return STANDARD_REWARD
    if iteration % milestone_period:
        return STANDARD_REWARD
    return EXPLOITER_CYCLE[(iteration // milestone_period - 1) % len(EXPLOITER_CYCLE)]


def reward_mode_name(mode):
    return {STANDARD_REWARD: "standard", ALTITUDE_HUNT_REWARD: "altitude_hunt",
            ATTACK_REWARD: "attack", DEFENSE_REWARD: "defense"}[validate_reward_mode(mode)]


def damage_reward(mode, dealt, taken, scale, own_weight=1.0):
    """Scalar/Torch reference. Does NOT scale geometry or any terminal reward."""
    mode = validate_reward_mode(mode)
    if mode == ATTACK_REWARD:
        return dealt * scale
    if mode == DEFENSE_REWARD:
        return -(taken * own_weight) * scale
    # Keep the exact former arithmetic for standard and altitude hunter.
    return (dealt - taken * own_weight) * scale
