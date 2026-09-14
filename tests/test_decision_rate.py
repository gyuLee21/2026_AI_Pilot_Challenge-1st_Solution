"""Catch clock acceleration, shrinking history and unequal A/B observations."""
import numpy as np
import pytest


def states(t=0.):
    a = np.zeros(16); b = np.zeros(16)
    a[:3] = [300*t, 0, -3000]; b[:3] = [300*t+600, 0, -3000]
    a[6] = b[6] = 300
    return a, b


def implementation():
    import importlib.util
    assert importlib.util.find_spec('submission.decision_rate') is not None, 'rate-aware adapter missing'
    from submission.decision_rate import RateObservation
    return RateObservation


def test_elapsed_time_and_hp_use_physical_seconds():
    r = implementation()()
    for tick in range(61):
        r.observe(*states(tick/60))
        r.record_command([0, 0, 0, 1])
    assert r.rec.t_sec == pytest.approx(1.)
    from claude_code.my_observation import damage_rate, METER_TO_FEET
    assert r.rec.hp_tgt == pytest.approx(1-damage_rate(600*METER_TO_FEET, 0, 0), abs=1e-7)


def test_history_lags_are_one_tenth_second_not_one_frame():
    r = implementation()()
    for tick in range(31):
        obs = r.observe(*states(tick/60))
        r.record_command([tick/60, 0, 0, .75])
    np.testing.assert_allclose(obs[-20:].reshape(5,4)[:,0], [.4,.3,.2,.1,0], atol=1e-7)
    np.testing.assert_allclose(obs[-20:].reshape(5,4)[:,3], [.75]*5)


def test_reset_has_no_time_damage_or_action_from_previous_episode():
    r = implementation()()
    for tick in range(9):
        r.observe(*states(tick/60)); r.record_command([1, 0, 0, 1])
    r.reset(); x = r.observe(*states())
    assert r.rec.t_sec == 0 and r.rec.hp_own == 1 and r.rec.hp_tgt == 1
    np.testing.assert_array_equal(x[-20:], np.zeros(20))


def test_angular_velocity_is_not_six_times_wrong():
    r = implementation()()
    for tick in range(13):
        a,b = states(tick/60); a[5] = 30*tick/60
        r.observe(a,b); r.record_command([0,0,0,1])
    assert r.rec.own_pqr_est[2] == pytest.approx(np.pi/6, abs=1e-7)
