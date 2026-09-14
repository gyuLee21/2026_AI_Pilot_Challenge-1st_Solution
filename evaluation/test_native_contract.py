"""CPU provider must preserve GPU command history and reset timing."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import native_baselines as native

native.setup()


class Actor:
    num_bins = 21
    def initial_state(self, *args): return None
    def act(self, obs, state, episode_start, sample=False):
        return torch.tensor([[10, 10, 10, 5]]), None


class ContractTests(unittest.TestCase):
    def provider(self, kind='checkpoint'):
        with patch.object(native, 'load_actor', return_value=(Actor(), None)):
            return native.rl_provider({'kind': kind})

    def context(self):
        a = np.zeros(51); a[2] = -4572; a[5] = 90; a[6] = 200
        b = a.copy(); b[0] = 700; b[5] = 270
        return SimpleNamespace(ownship_state=a, target_state=b)

    def test_reset_does_not_integrate_future_time(self):
        p = self.provider(); p.compute_action(self.context())
        self.assertEqual(p.recon.t_sec, 0.)
        for _ in range(6): p.compute_action(self.context())
        self.assertAlmostEqual(p.recon.t_sec, .1)
        p.reset(); p.compute_action(self.context())
        self.assertEqual(p.recon.t_sec, 0.)

    def test_gpu_history_uses_applied_throttle(self):
        p = self.provider(); result = p.compute_action(self.context())
        self.assertAlmostEqual(float(result.action[3]), .25)
        self.assertAlmostEqual(float(p.recon.action_history[0,3]), .25)

    def test_legacy_history_keeps_raw_throttle(self):
        p = self.provider('legacy184_bundle'); p.compute_action(self.context())
        self.assertAlmostEqual(float(p.recon.action_history[0,3]), -.5)

    def test_cutoff_requests_every_physics_frame(self):
        from pathlib import Path
        with patch('cutoff_udp_provider.CutoffUDPActionProvider', side_effect=lambda *a, **kw: kw):
            config = native.baseline_provider('cutoff', Path('unused'))
        self.assertEqual(config['action_repeat'], 1)

    def test_cutoff_packet_uses_up_positive_altitude(self):
        import cutoff_udp_provider as c
        a = self.context().ownship_state
        packet = c.PLANE_INFO_STRUCT.unpack(c._pack_plane_info(0,1,a))
        self.assertEqual(packet[5], 4572.)
        init = c.INIT_STRUCT.unpack(c._pack_init(a,a))
        self.assertEqual(init[3], 4572.)
        self.assertEqual(init[10], 4572.)

    def test_cutoff_10hz_repeats_six_physics_frames(self):
        from pathlib import Path
        with patch('cutoff_udp_provider.CutoffUDPActionProvider', side_effect=lambda *a, **kw: kw):
            config = native.baseline_provider('cutoff_10hz', Path('unused'))
        self.assertEqual(config['action_repeat'], 6)


if __name__ == '__main__': unittest.main()
