"""CPU-only production submission parity, legacy isolation, and hybrid handoff."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from claude_code import my_observation as MO
from claude_code.action_provider import MLPActionProvider
from claude_code.altguard_provider import AltGuardMPCProvider
from claude_code.altblend_provider import AltBlendMPCProvider
from claude_code.model import make_actor_critic, save_bundle, make_obs_normalizer
from claude_code.observation_contract import (
    CUDAObservationState, CUDA_OBSERVATION_CONTRACT, uses_cuda_observation_contract,
)
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.tests.submission_contract_audit import (
    CapturingProvider, ShadowActor, _states, _context, _command_array,
)
from dogfight.ai.action_provider import ActionContext, ActionResult
from dogfight.unreal import ProviderCommandPolicy


class FakeRates:
    def reset(self):
        pass

    def update(self, attitude, time):
        return np.zeros(3)


class FakeMPC:
    def __init__(self):
        self.reset()

    def reset(self, context=None):
        self.calls = 0

    def compute_action(self, context):
        self.calls += 1
        command = np.array([.1, -.15, .05, .25 + .01 * (self.calls % 40)], dtype=np.float32)
        return ActionResult(command, "mock_mpc")

    def close(self):
        pass


class SubmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="aip_submission_contract_")
        cls.bundles = {}
        for kind, gru in (("mlp", 0), ("gru", 16)):
            torch.manual_seed(67)
            model = make_actor_critic(obs_dim=214, act_dim=4, num_bins=21,
                                     hidden=(32, 32, 32), gru_size=gru)
            norm = {"mean": np.linspace(-.1, .1, 214).tolist(),
                    "var": np.linspace(.3, 1.3, 214).tolist(), "count": 100.}
            for contract in ("explicit", "old_cuda", "legacy_cpu"):
                path = Path(cls.temp.name) / f"{kind}_{contract}"
                metadata = {"observation_module": "claude_code.my_observation",
                            "observation_mode": MO.OBSERVATION_MODE}
                if contract != "legacy_cpu":
                    metadata["trainer"] = "cuda_fdm.PPOGPUTrainer"
                if contract == "explicit":
                    metadata.update(observation_contract=CUDA_OBSERVATION_CONTRACT,
                                    policy_hz=10, action_repeat=6)
                save_bundle(model, path, obs_norm=norm, extra_metadata=metadata)
                cls.bundles[kind, contract] = path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_basic(self, kind, contract="explicit"):
        provider = CapturingProvider(self.bundles[kind, contract], stochastic=False, debug_obs=False)
        policy = ProviderCommandPolicy(provider, observation_mode=MO.OBSERVATION_MODE,
                                       observation_fn=MO.build_observation, action_repeat=6)
        policy.reset(None)
        return provider, policy

    def test_actual_basic_mlp_gru_and_unversioned_cuda_bundles(self):
        for kind in ("mlp", "gru"):
            for contract in ("explicit", "old_cuda"):
                with self.subTest(kind=kind, contract=contract):
                    provider, policy = self.make_basic(kind, contract)
                    norm = make_obs_normalizer(provider.metadata["obs_normalization"])
                    for episode in range(2):
                        policy.reset(None)
                        reference = BatchObsReward(1, device="cpu", enable_kernel=False)
                        shadow = ShadowActor(provider.model)
                        last = None
                        for k in range(24):
                            states = _states("maneuver" if episode == 0 else "cone_transition", k)
                            if k:
                                reference.push_actions(torch.tensor(np.stack([last, last])))
                                reference.advance(torch.tensor(states))
                            expected = reference.build_obs(torch.tensor(states))[0].numpy()
                            command = _command_array(policy.compute_command(_context(states, k * 6)))
                            np.testing.assert_allclose(provider.captured_obs, expected, atol=3e-6, rtol=2e-5)
                            np.testing.assert_allclose(norm(provider.captured_obs), norm(expected), atol=2e-5)
                            _, expected_command, _ = shadow.predict(norm(expected))
                            np.testing.assert_array_equal(command, expected_command)
                            for frame in range(1, 6):
                                cached = _command_array(policy.compute_command(_context(states, k * 6 + frame)))
                                np.testing.assert_array_equal(cached, command)
                            self.assertAlmostEqual(provider._cuda_observation.rec.t_sec, k * .1)
                            last = command

    def test_legacy_cpu_keeps_preexisting_context_and_raw_history(self):
        provider = MLPActionProvider(self.bundles["mlp", "legacy_cpu"], stochastic=False, debug_obs=False)
        provider.reset()
        states = _states("maneuver", 0)
        observation = MO.build_observation(states[0], states[1], MO.get_reconstructor()._geo)
        result = provider.compute_action(ActionContext(None, None, states[0], states[1], observation))
        self.assertIsNone(provider._cuda_observation)
        self.assertFalse(provider.builds_own_observation)
        self.assertAlmostEqual(MO.get_reconstructor().t_sec, .1)
        expected = result.action.astype(np.float64).copy()
        expected[3] = 2 * expected[3] - 1
        np.testing.assert_allclose(MO.get_reconstructor().action_history[0], expected, atol=2e-7)

    def test_agents_have_private_reconstruction_and_reset(self):
        p1, policy1 = self.make_basic("gru")
        p2, policy2 = self.make_basic("gru")
        for k in range(5):
            states = _states("maneuver", k)
            for frame in range(6):
                a = _command_array(policy1.compute_command(_context(states, k * 6 + frame)))
                b = _command_array(policy2.compute_command(_context(states, k * 6 + frame)))
                np.testing.assert_array_equal(a, b)
        old_time = p2._cuda_observation.rec.t_sec
        old_hist = p2._cuda_observation.rec.action_history.copy()
        policy1.reset(None)
        self.assertEqual(p1._cuda_observation.rec.t_sec, 0.)
        self.assertEqual(p2._cuda_observation.rec.t_sec, old_time)
        np.testing.assert_array_equal(p2._cuda_observation.rec.action_history, old_hist)

    def test_actual_entrypoints_construct_private_contract_without_network(self):
        from claude_code import submission, submission_client, build_submission
        self.assertIn("claude_code.observation_contract", build_submission.HIDDEN_IMPORTS)
        constructed = []
        class NoSocketClient:
            def __init__(self, **kwargs):
                self.policy = kwargs["command_policy"]
                constructed.append(self.policy)
            def run(self):
                self.policy.reset(None)
                for k in range(18):
                    command = self.policy.compute_command(_context(_states("maneuver", k // 6), k))
                    assert np.isfinite(_command_array(command)).all()
                assert abs(self.policy.action_provider._cuda_observation.rec.t_sec - .2) < 1e-9
        with patch.object(submission, "BUNDLE_DIR", str(self.bundles["gru", "explicit"])), \
                patch.object(submission, "UnrealAIPilotUDPClient", NoSocketClient):
            submission.main()
        for mode in ("basic", "altguard", "altblend"):
            cfg = dict(server_ip="127.0.0.1", team_name="local-contract-test", mode=mode,
                       bundle_dir=str(self.bundles["mlp", "explicit"]), mpc_root=self.temp.name)
            with self.subTest(mode=mode), patch.object(sys, "argv", ["submission_client"]), \
                    patch.object(submission_client, "load_config", return_value=cfg), \
                    patch("dogfight.unreal.UnrealAIPilotUDPClient", NoSocketClient), \
                    patch("claude_code.altguard_provider._load_teamshare_mpc", return_value=(FakeMPC(), None)), \
                    patch("claude_code.altblend_provider._load_teamshare_mpc", return_value=(FakeMPC(), None)), \
                    patch.dict(sys.modules, {"mpc.transforms": SimpleNamespace(AngularRateEstimator=FakeRates)}):
                submission_client.main()
        self.assertEqual(len(constructed), 4)

    def make_hybrid(self, provider_type, kind):
        module = "claude_code.altguard_provider" if provider_type is AltGuardMPCProvider else "claude_code.altblend_provider"
        transforms = SimpleNamespace(AngularRateEstimator=FakeRates)
        with patch(module + "._load_teamshare_mpc", return_value=(FakeMPC(), None)), \
                patch.dict(sys.modules, {"mpc.transforms": transforms}):
            provider = provider_type(self.bundles[kind, "explicit"], stochastic=False)
        policy = ProviderCommandPolicy(provider, observation_fn=lambda *a: np.zeros(16), action_repeat=1)
        policy.reset(None)
        return provider, policy

    def test_hybrid_modes_match_observation_and_latest_applied_command(self):
        for kind in ("mlp", "gru"):
            for provider_type in (AltGuardMPCProvider, AltBlendMPCProvider):
                with self.subTest(kind=kind, mode=provider_type.__name__):
                    provider, policy = self.make_hybrid(provider_type, kind)
                    for episode in range(2):
                        policy.reset(None)
                        reference = BatchObsReward(1, device="cpu", enable_kernel=False)
                        last = None
                        for k in range(20):
                            states = _states("maneuver", k)
                            states[0, 2] = -([1800., 500., 900., 1800.][k // 5])
                            if k:
                                reference.push_actions(torch.tensor(np.stack([last, last])))
                                reference.advance(torch.tensor(states))
                            expected = reference.build_obs(torch.tensor(states))[0].numpy()
                            for frame in range(6):
                                last = _command_array(policy.compute_command(_context(states, k * 6 + frame)))
                                if frame == 0:
                                    observed = MO.build_observation(states[0], states[1], provider._geo,
                                                                    reconstructor=provider._rec)
                                    np.testing.assert_allclose(observed, expected, atol=3e-6, rtol=2e-5)
                            self.assertAlmostEqual(provider._rec.t_sec, k * .1)
                        self.assertGreater(provider.mpc.calls, 0)

    def test_cuda_cadence_mismatch_is_rejected(self):
        provider, _ = self.make_basic("mlp")
        with self.assertRaisesRegex(ValueError, "action_repeat=6"):
            ProviderCommandPolicy(provider, action_repeat=3)
        hybrid, _ = self.make_hybrid(AltBlendMPCProvider, "mlp")
        with self.assertRaisesRegex(ValueError, "action_repeat=1"):
            ProviderCommandPolicy(hybrid, action_repeat=6)

    def test_contract_rejects_unknown_version_and_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            uses_cuda_observation_contract({"observation_contract": "future_unknown"})
        adapter = CUDAObservationState()
        states = _states("stationary", 0)
        states[0, 6] = float("nan")
        with self.assertRaises(FloatingPointError):
            adapter.observation(*states)
        with self.assertRaises(FloatingPointError):
            adapter.record_command([0, 0, 0, float("inf")])
        with self.assertRaises(ValueError):
            adapter.record_command([0, 0, 0, -.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
