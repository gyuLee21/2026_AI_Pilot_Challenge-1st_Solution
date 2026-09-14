"""Protect 10Hz inference/60Hz replies and reset against packaging regressions."""
import importlib.util
from pathlib import Path
import numpy as np


def test_main_policy_preserves_evaluation_actions_and_repeats_between_decisions():
    assert importlib.util.find_spec('submission.main_client') is not None, 'main entry point missing'
    from submission.main_client import MainPolicy
    from submission.frozen_policy import load
    from submission.decision_rate import DecisionRateProvider
    from dogfight.ai.action_provider import ActionContext
    from dogfight.unreal.client import RemoteClientContext, PlaneSnapshot
    from dogfight.unreal import protocol as wire
    root = Path(__file__).resolve().parents[1]
    actor = load(root/'artifacts/evaluations/transfer_36000_20260913/cpu_36k_10hz_vs_60hz/model.pt')
    policy = MainPolicy(actor)
    reference = DecisionRateProvider(actor,None,10)
    for episode in range(2):
        policy.reset(None); reference.reset()
        actions=[]
        for frame in range(60):
            own=wire.PlaneInfo(frame,episode,wire.Vector3D(frame*5,0,4000),
                wire.Rotation3D(0,frame*.05,0),wire.Vector3D(300,0,0))
            opp=wire.PlaneInfo(frame,1-episode,wire.Vector3D(800-frame*5,100,4000),
                wire.Rotation3D(0,0,180),wire.Vector3D(300,0,0))
            context=RemoteClientContext(plane_id=episode,frame_index=frame,
                own_plane=PlaneSnapshot(True,episode,frame,own),
                enemy_plane=PlaneSnapshot(True,1-episode,frame,opp))
            cmd=policy.compute_command(context)
            action=[cmd.roll_cmd,cmd.pitch_cmd,cmd.yaw_cmd,cmd.throttle_cmd]
            expected=reference.compute_action(ActionContext(None,None,
                np.array([frame*5,0,-4000,0,frame*.05,0,300,0,0],dtype=np.float32),
                np.array([800-frame*5,100,-4000,0,0,180,300,0,0],dtype=np.float32),None,{})).action
            np.testing.assert_array_equal(action,expected)
            actions.append(action)
            if frame%6: np.testing.assert_array_equal(actions[-1],actions[-2])
        assert policy.provider.calls == 60
        assert policy.provider.decisions == 10
        assert policy.provider.observation.rec.t_sec == reference.observation.rec.t_sec
