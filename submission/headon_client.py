"""Competition entry point: frozen 46k actor, one decision per 60Hz state pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext
from dogfight.unreal.client import UnrealAIPilotUDPClient
from dogfight.unreal.protocol import CMD
from submission.decision_rate import DecisionRateProvider
from submission.frozen_policy import load

TEAM = "추락도락이다_HeadOn"
SOURCE_SHA256 = "858d4ee83d5cc3cd073ebf804cc8d29358ad047adfb991692cccc5f858a2faa3"


def plane_state(plane):
    # Same packet contract as dogfight.unreal.policies.plane_info_to_state.
    return np.array([plane.position.x, plane.position.y, -plane.position.z,
                     plane.rotation.roll, plane.rotation.pitch, plane.rotation.yaw,
                     plane.velocity.x, plane.velocity.y, plane.velocity.z], dtype=np.float32)


class HeadonPolicy:
    def __init__(self, actor):
        self.provider = DecisionRateProvider(actor, None, 60)

    def reset(self, context):
        self.provider.reset()

    def compute_command(self, context):
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(context.plane_id, context.frame_index, 0., 0., 0., 1.)
        action = self.provider.compute_action(ActionContext(
            sim=None, opponent_sim=None,
            ownship_state=plane_state(context.own_plane.plane_info),
            target_state=plane_state(context.enemy_plane.plane_info),
            observation=None, info={"frame_index": context.frame_index})).action
        return CMD(context.plane_id, context.frame_index, *map(float, action))


def self_test(actor):
    provider = DecisionRateProvider(actor, None, 60)
    own = np.array([0, 0, -5000, 0, 0, 0, 300, 0, 0], dtype=np.float32)
    enemy = np.array([5539, 0, -5000, 0, 0, 180, 300, 0, 0], dtype=np.float32)
    for _ in range(600):
        provider.compute_action(ActionContext(None, None, own, enemy, None, {}))
    assert provider.calls == provider.decisions == 600
    assert not torch.cuda.is_initialized()
    result = dict(ok=True, device="cpu", iteration=46000, hz=60,
                  observation_size=actor.obs_dim, team_name=TEAM,
                  p50_ms=float(np.percentile(provider.latencies[30:], 50)*1000),
                  p99_ms=float(np.percentile(provider.latencies[30:], 99)*1000),
                  max_ms=float(max(provider.latencies[30:])*1000))
    assert result["p99_ms"] < 1000/60, result
    print(json.dumps(result, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen = getattr(sys, "frozen", False)
    resources = Path(sys._MEIPASS) if frozen else Path(__file__).resolve().parent
    config_dir = Path(sys.executable).parent if frozen else resources
    cfg = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
    expected = dict(server_ip="127.0.0.1", server_port=9999, team_name=TEAM,
                    decision_hz=60, device="cpu")
    if cfg != expected:
        raise ValueError("config.json must match the competition head-on configuration")
    payload_path = resources / "headon46k.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload["source_sha256"] != SOURCE_SHA256 or payload["iteration"] != 46000:
        raise ValueError("Unexpected model checkpoint")
    actor = load(payload_path)
    if args.self_test:
        self_test(actor)
        return
    # Warm up before connecting; weights and normalization stay resident in RAM.
    actor.logits(torch.zeros(1, 214))
    policy = HeadonPolicy(actor)
    print(f"Head-on 46000 | CPU | 60Hz | {TEAM} | 127.0.0.1:9999", flush=True)
    client = UnrealAIPilotUDPClient(policy, server_ip=cfg["server_ip"],
        server_port=cfg["server_port"], team_name=TEAM, command_delay_sec=0.,
        enable_terminal_monitor=False)
    try:
        client.run()
    except KeyboardInterrupt:
        client.stop()


if __name__ == "__main__":
    main()
