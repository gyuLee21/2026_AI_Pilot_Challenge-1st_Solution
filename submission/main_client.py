"""36.5k main submission: 10Hz argmax decisions, replies on every 60Hz state pair."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from dogfight.ai.action_provider import ActionContext
from dogfight.unreal.client import UnrealAIPilotUDPClient
from submission.headon_client import HeadonPolicy
from submission.decision_rate import DecisionRateProvider
from submission.frozen_policy import load

TEAM = '추락도락이다'
SOURCE_SHA256 = '915998954da2a287767231029850f645f47718aa6e7763f5d6fb1e5284516a7b'


class MainPolicy(HeadonPolicy):
    def __init__(self, actor):
        # Reuse precisely the existing packet mapping and reset behavior.
        self.provider = DecisionRateProvider(actor, None, 10)


def self_test(actor):
    provider = DecisionRateProvider(actor, None, 10)
    own = np.array([0,0,-5000,0,0,0,300,0,0],dtype=np.float32)
    enemy = np.array([5539,0,-5000,0,0,180,300,0,0],dtype=np.float32)
    for frame in range(600):
        result = provider.compute_action(ActionContext(None,None,own,enemy,None,{}))
        assert np.isfinite(result.action).all()
    assert provider.calls == 600 and provider.decisions == 100
    assert not torch.cuda.is_initialized()
    report = dict(ok=True,device='cpu',iteration=36500,hz=10,response_hz=60,
        calls=provider.calls,decisions=provider.decisions,observation_size=actor.obs_dim,team_name=TEAM,
        p50_ms=float(np.percentile(provider.latencies[30:],50)*1000),
        p99_ms=float(np.percentile(provider.latencies[30:],99)*1000),
        max_ms=float(max(provider.latencies[30:])*1000))
    assert report['p99_ms'] < 1000/60, report
    print(json.dumps(report,ensure_ascii=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    frozen=getattr(sys,'frozen',False)
    resources=Path(sys._MEIPASS) if frozen else Path(__file__).resolve().parent
    config_dir=Path(sys.executable).parent if frozen else resources
    cfg=json.loads((config_dir/'config.json').read_text(encoding='utf-8'))
    expected=dict(server_ip='127.0.0.1',server_port=9999,team_name=TEAM,decision_hz=10,device='cpu')
    if cfg != expected:
        raise ValueError('config.json must match the competition main configuration')
    payload_path=resources/'main_policy.pt'
    payload=torch.load(payload_path,map_location='cpu',weights_only=True)
    if payload['source_sha256'] != SOURCE_SHA256 or payload['iteration'] != 36500:
        raise ValueError('Unexpected model checkpoint')
    actor=load(payload_path)
    if args.self_test:
        self_test(actor)
        return
    actor.logits(torch.zeros(1,214))
    print(f'Main 36500 | CPU | 10Hz decisions / 60Hz replies | {TEAM} | 127.0.0.1:9999',flush=True)
    client=UnrealAIPilotUDPClient(MainPolicy(actor),server_ip=cfg['server_ip'],
        server_port=cfg['server_port'],team_name=TEAM,command_delay_sec=0.,enable_terminal_monitor=False)
    try:
        client.run()
    except KeyboardInterrupt:
        client.stop()


if __name__=='__main__': main()
