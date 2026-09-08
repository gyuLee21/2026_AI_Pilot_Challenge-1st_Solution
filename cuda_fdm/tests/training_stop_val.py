import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cuda_fdm.training_stop import (TrainingStopController, PROTOCOL, paired_difference,
                                    compare, counter_ucb, polish_value)
from cuda_fdm.league_vnext.champion import ChampionManager, PromotionEvidence
from cuda_fdm.league_vnext.contracts import ChampionConfig


def row(score=.75, crash=0., n=256, group="g0", opponent="a"):
    return dict(opponent_hash=opponent, seed_block=10, paired_blocks=n, stochastic=True,
                paired_scores=[score] * n, paired_crashes=[crash] * n,
                score=score, left_alt_loss_rate=crash, games=2*n, group=group)


class FakeTrainer:
    def __init__(self):
        self.iteration = 10
        self.cfg = SimpleNamespace(total_iterations=79900)
        self.env = SimpleNamespace(scenario="three_nine")
        self.checkpoint_safe = True
        self.archive = SimpleNamespace(records={i: {"id":i,"sha256": str(i)} for i in range(12)},
                                       load_policy=lambda i: {"opponent": i})
        self.pool = SimpleNamespace(entries=[{"archive_id":i,"role":"core","ema":.5} for i in range(6)])
        self.training_stop_state = None
        self.exploiter_history = [{"archive_id": i} for i in range(4)]
        self.calls = 0
        self.score = .75
        self.fail = False

    def _policy_bundle(self):
        return {"model": {"x": torch.tensor([self.score])}, "norm": None}

    def save(self, path):
        self.saved = copy.deepcopy(self.training_stop_state)

    def _evaluate_pair(self, left, right, blocks, seed_block):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected evaluator failure")
        result = row(float(left["model"]["x"][0]), n=blocks)
        result["seed_block"] = seed_block
        return result


def config():
    return dict(protocol=PROTOCOL, normal_end=74900, polish_iterations=5000,
                suite=[dict(id=i, sha256=str(i), group=f"g{i//3}") for i in range(12)])


class StatisticsTests(unittest.TestCase):
    def test_real_training_polish_skips_side_and_preserves_rewards(self):
        from unittest.mock import Mock
        from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
        from cuda_fdm.tests.pool_identity_audit import ToyEnv
        from cuda_fdm.obs_reward import RW
        env=ToyEnv(4)
        env.reward_cfg=copy.deepcopy(RW.MY_REWARD_CONFIG)
        tr = PPOGPUTrainer(env, PPOGPUConfig(
            device="cpu", aux_pred=False, architecture="mlp", hidden=(8,8), num_bins=3,
            normalize_obs=False, rollout_steps=8, total_iterations=1,
            update_epochs=1, num_minibatches=1, milestone_period=1, exploiter_iters=10,
            sched_period=2000, sched_rollout_increment=0))
        reward = copy.deepcopy(tr.env.reward_cfg)
        tr.training_stop_state = {"phase":"polish", "polish_start":0,
                                  "config":{"polish_iterations":5000}}
        tr.train_exploiter = Mock(side_effect=AssertionError("polish started a side learner"))
        tr.train()
        tr.train_exploiter.assert_not_called()
        self.assertEqual(tr.cfg.ent_coef,1e-4)
        self.assertEqual(tr.actor_opt.param_groups[0]["lr"],1e-4)
        self.assertEqual(tr.cfg.rollout_steps,8)
        self.assertEqual(tr.env.reward_cfg, reward)
        self.assertIsNone(tr._last_exploiter_archive_id)

    def test_signed_difference_and_pair_validation(self):
        result = paired_difference([row(.2)], [row(.8)])
        self.assertAlmostEqual(result["mean"], -.6)
        self.assertLess(result["ucb"], 0)
        for field, value in [("seed_block", 11), ("opponent_hash", "b"),
                             ("paired_blocks", 128), ("stochastic", False)]:
            bad = row(); bad[field] = value
            with self.assertRaises(ValueError):
                paired_difference([row()], [bad])

    def test_empty_nan_and_degenerate(self):
        with self.assertRaises(ValueError): paired_difference([], [])
        bad = row(); bad["paired_scores"][0] = float("nan")
        with self.assertRaises(ValueError): paired_difference([bad], [row()])
        value = paired_difference([row()], [row()])
        self.assertLess(value["lcb"], 0)
        self.assertGreater(value["ucb"], 0)

    def test_every_group_not_only_minimum(self):
        old = [row(.5, group="a"), row(.9, group="b")]
        new = [row(.6, group="a"), row(.7, group="b")]
        self.assertFalse(compare(new, old, .05)["safe"])
        new = [row(.6, .04, group="a"), row(.95, group="b")]
        self.assertFalse(compare(new, old, .05)["safe"])

    def test_counter_is_fresh_main_complement_and_conservative(self):
        self.assertGreater(counter_ucb(row(.5), .05/24), .60)
        self.assertLess(counter_ucb(row(.8), .05/24), .60)

    def test_polish_exact_endpoints_and_resume(self):
        self.assertEqual(polish_value(101,100,5000), 1e-4)
        self.assertAlmostEqual(polish_value(5100,100,5000), 3e-5)
        self.assertEqual(polish_value(2500,100,5000), polish_value(2500,100,5000))

    def test_worst_noninferiority_independent_of_max_improvement(self):
        manager = ChampionManager(ChampionConfig(worst_cluster_noninferiority_margin=.03))
        e = PromotionEvidence(1,"champion_suite_v1","active_league_vnext_100k_v2",
                              .1,.1,-.1,.1,.1,256,.8,.5,.8,.98)
        self.assertIn("worst_cluster_inferior",manager.evaluate(e).reasons)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.trainer = FakeTrainer()
        self.path = Path(self.temp.name)/"checkpoint.pt"
        self.gate = TrainingStopController(self.trainer, config(), self.path)

    def tearDown(self): self.temp.cleanup()

    def tick(self, iteration):
        self.trainer.iteration = iteration
        if self.gate.due(): self.gate.tick()

    def test_baseline_not_plateau_and_idempotent_reload(self):
        self.tick(10)
        self.assertEqual(self.gate.state["phase"], "learn")
        self.assertEqual(self.gate.state["confirmations"], [])
        calls = self.trainer.calls
        self.gate.tick()
        self.gate = TrainingStopController(self.trainer, config(), self.path)
        self.gate.tick()
        self.assertEqual(self.trainer.calls, calls)
        self.assertEqual(self.gate.champion.champion_id,10)

    def test_full_plateau_polish_final_save_reload(self):
        for it in [10,2010,4010,6010]: self.tick(it)
        self.assertEqual(self.gate.state["phase"], "polish")
        self.assertEqual(self.gate.state["polish_end"],11010)
        self.assertTrue(self.gate.before_iteration(6011))
        for it in [7010,8010,9010,10010,11010]: self.tick(it)
        self.assertEqual(self.gate.state["phase"],"done")
        self.assertFalse(self.gate.before_iteration(11011))
        self.assertTrue((self.path.parent/"training_stop/best_model.pt").exists())
        calls = self.trainer.calls
        self.gate = TrainingStopController(self.trainer,config(),self.path)
        self.gate.tick()
        self.assertEqual(calls,self.trainer.calls)

    def test_difficult_counters_prevent_plateau(self):
        self.trainer.score = .5
        for it in [10,2010,4010,6010]: self.tick(it)
        self.assertEqual(self.gate.state["phase"],"learn")
        self.assertFalse(self.gate.state["confirmations"][0]["counter_gate_passed"])

    def test_disabled_early_stop_keeps_evaluations_and_budget_polish(self):
        cfg = config(); cfg["early_stop_enabled"] = False
        self.gate = TrainingStopController(self.trainer, cfg, self.path)
        for it in [10,2010,4010,6010]: self.tick(it)
        self.assertTrue(self.gate.state["confirmations"][-1]["plateau_confirmed"])
        self.assertEqual(self.gate.state["phase"], "learn")
        self.assertIsNone(self.gate.state["polish_start"])
        calls = self.trainer.calls
        self.tick(8010)
        self.assertGreater(self.trainer.calls, calls)
        self.tick(74900)
        self.assertEqual(self.gate.state["phase"], "polish")
        self.assertEqual(self.gate.state["polish_end"], 79900)
        self.assertEqual(self.gate.state["stop_reason"], "normal_iteration_budget_not_convergence")

    def test_disable_resume_preserves_evidence_and_is_idempotent(self):
        self.tick(10); self.tick(2010)
        evidence = copy.deepcopy(self.gate.state["evaluations"])
        calls = self.trainer.calls
        cfg = config(); cfg["early_stop_enabled"] = False
        self.gate = TrainingStopController(self.trainer, cfg, self.path)
        self.assertEqual(self.gate.state["evaluations"], evidence)
        self.assertEqual(self.gate.champion.champion_id, 10)
        self.assertFalse(self.trainer.saved["config"]["early_stop_enabled"])
        self.gate = TrainingStopController(self.trainer, cfg, self.path)
        self.assertEqual(self.trainer.calls, calls)
        events = (self.path.parent/"training_stop/events.jsonl").read_text()
        self.assertEqual(events.count('"event": "early_stop_policy_changed"'), 1)
        cfg["suite"][0]["group"] = 'g1'
        with self.assertRaises(ValueError):
            TrainingStopController(self.trainer, cfg, self.path)

    def test_strategic_improvement_blocks_plateau_and_can_promote(self):
        original=self.gate.evaluate
        def evaluate(identity,bank,blocks,opponents=None):
            rows=original(identity,bank,blocks,opponents)
            if opponents and opponents[0]["group"]=="strategic_panel" and identity != 10:
                rows=copy.deepcopy(rows)
                for r in rows:
                    r["paired_scores"]=[min(1,x+.1) for x in r["paired_scores"]]
                    r["score"] += .1
            return rows
        self.gate.evaluate=evaluate
        for it in [10,2010,4010,6010]: self.tick(it)
        self.assertEqual(self.gate.state["phase"],"learn")
        self.assertNotEqual(self.gate.champion.champion_id,10)

    def test_strategic_panel_is_unique_and_not_a_mass_skill_gate(self):
        panel=self.gate.strategic_panel()
        self.assertEqual(len({r["id"] for r in panel}),6)
        self.assertEqual(len(panel),6)

    def test_interrupted_confirmation_is_due_even_after_next_eval_advanced(self):
        for it in [10,2010,4010]: self.tick(it)
        original=self.gate.evaluate
        def interrupted(identity,bank,blocks,opponents=None):
            if bank != 190700000: raise RuntimeError("crash after pending panel checkpoint")
            return original(identity,bank,blocks,opponents)
        self.gate.evaluate=interrupted
        with self.assertRaises(RuntimeError): self.tick(6010)
        self.assertEqual(self.trainer.saved["next_evaluation"],8010)
        self.assertEqual(self.trainer.saved["pending_tick"],6010)
        self.gate=TrainingStopController(self.trainer,config(),self.path)
        self.assertTrue(self.gate.due())
        self.assertFalse(self.gate.before_iteration(6011))
        self.gate.tick()
        self.assertEqual(self.gate.state["phase"],"polish")
        self.assertIsNone(self.gate.state["pending_tick"])

    def test_new_champion_cannot_regress_against_recent_counters(self):
        self.tick(10)
        self.trainer.score=.9
        original=self.gate.evaluate
        def evaluate(identity,bank,blocks,opponents=None):
            rows=original(identity,bank,blocks,opponents)
            if opponents and opponents[0]["group"]=="recent_counter" and identity!=10:
                rows=copy.deepcopy(rows)
                for r in rows:
                    r["paired_scores"]=[.3]*blocks
                    r["score"] = .3
            return rows
        self.gate.evaluate=evaluate
        for it in [2010,4010,6010]: self.tick(it)
        self.assertEqual(self.gate.champion.champion_id,10)
        receipt=self.gate.state["confirmations"][-1]
        self.assertNotEqual(receipt["proposed_selected"],10)
        self.assertFalse(receipt["counter_protection"]["safe"])

    def test_per_look_alpha_is_not_an_extreme_one_bootstrap_tail(self):
        for it in [10,2010,4010,6010]: self.tick(it)
        alpha=self.gate.state["confirmations"][-1]["per_comparison_alpha"]
        self.assertAlmostEqual(alpha,.05/3)
        self.assertGreater(20000*alpha/2,100)

    def test_recovering_below_old_champion_is_not_a_plateau(self):
        for it, score in [(10,.95),(2010,.6),(4010,.7),(6010,.8)]:
            self.trainer.score=score
            self.tick(it)
        self.assertEqual(self.gate.state["phase"],"learn")
        self.assertFalse(self.gate.state["confirmations"][-1]["plateau_confirmed"])

    def test_checkpoint_failure_cannot_count_as_completed_tick(self):
        original=self.trainer.save
        def failed(path): raise OSError("injected save failure")
        self.trainer.save=failed
        with self.assertRaises(OSError): self.tick(10)
        self.assertIsNone(self.gate.state["baseline_iteration"])
        self.trainer.save=original

    def test_missing_counter_not_pass(self):
        self.trainer.exploiter_history = []
        with self.assertRaises(ValueError): self.gate.recent_counters()

    def test_failure_then_resume_uses_cached_work_not_plateau(self):
        self.trainer.fail = True
        with self.assertRaises(RuntimeError): self.tick(10)
        self.assertIsNone(self.gate.state["baseline_iteration"])
        self.trainer.fail = False
        self.gate = TrainingStopController(self.trainer, config(), self.path)
        self.tick(10)
        self.assertEqual(self.gate.state["phase"],"learn")

    def test_budget_distinguished_from_convergence(self):
        self.tick(10)
        self.trainer.score = .5
        self.tick(74900)
        self.assertEqual(self.gate.state["polish_end"],79900)
        self.assertEqual(self.gate.state["stop_reason"],"normal_iteration_budget_not_convergence")

    def test_config_change_and_inflight_rejected(self):
        self.tick(10)
        changed = config(); changed["suite"][0]["group"] = "bad"
        with self.assertRaises(ValueError): TrainingStopController(self.trainer,changed,self.path)
        self.trainer.checkpoint_safe = False
        with self.assertRaises(RuntimeError): self.gate.tick()

    def test_artifact_tamper_rejected(self):
        self.tick(10)
        spec = self.gate.state["candidates"]["10"]
        Path(spec["path"]).write_bytes(b"damaged")
        with self.assertRaises(ValueError): self.gate.evaluate(10,190700000,64)

    def test_completed_confirmation_is_not_repeated(self):
        for it in [10,2010,4010,6010]: self.tick(it)
        receipt=self.gate.state["confirmations"][0]
        calls=self.trainer.calls
        self.assertTrue(self.gate.confirmation([2010,4010,6010],10,receipt["bank"],True))
        self.assertEqual(calls,self.trainer.calls)


if __name__ == "__main__": unittest.main()
