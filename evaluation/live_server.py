"""Real organizer-server client. Raw capture is evidence, not inferred match scores.

Never hosts a replacement simulator or auto-labels a disconnect as a draw.
"""
from pathlib import Path
import argparse
import json
import struct
import time


class FramePairs:
    def __init__(self): self.clear()
    def clear(self): self.pending={}; self.last=-1
    def add(self, plane, frame, value):
        if plane not in (0,1) or frame<=self.last: return None
        row=self.pending.setdefault(frame,{})
        row[plane]=value
        if len(row)==2:
            self.last=frame
            self.pending={k:v for k,v in self.pending.items() if k>frame}
            return row[0],row[1]
        while len(self.pending)>8: del self.pending[min(self.pending)]
        return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--hz',type=int,choices=(10,60),default=60)
    args=ap.parse_args()
    spec=Path(args.spec).resolve(); out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    import native_baselines as native
    native.setup()
    import torch
    from tournament import prepare
    from submission.frozen_policy import export_checkpoint,FrozenPolicy
    from submission.decision_rate import DecisionRateProvider
    from dogfight.unreal.client import UnrealAIPilotUDPClient
    from dogfight.unreal.policies import ProviderCommandPolicy
    from dogfight.unreal.protocol import MessageType,unpack_plane_info
    from policy_io import load_bundle_policy
    from dataclasses import asdict
    models=prepare(spec)['models']
    model=next(m for m in models if m['name']==args.model)
    if model.get('kind')=='legacy184_bundle':
        actor,norm=load_bundle_policy(model['path'],'cpu',model.get('legacy_min_altitude_m',300))
    elif model['name']=='gylee_current30000':
        actor=FrozenPolicy(export_checkpoint(model['path'],out/'actor.pt')); norm=None
    else: actor,norm=native.load_actor(model)
    provider=DecisionRateProvider(actor,norm,args.hz)
    policy=ProviderCommandPolicy(provider,action_repeat=1)
    raw=(out/'packets.bin').open('xb')
    events=(out/'events.jsonl').open('x',encoding='utf-8')
    def event(**kw):
        events.write(json.dumps(dict(wall=time.time(),**kw),ensure_ascii=False)+'\n');events.flush()
    class CaptureClient(UnrealAIPilotUDPClient):
        def __init__(self):
            self.pairs=FramePairs();self.frames=0;self.episode=0
            super().__init__(policy,server_ip='127.0.0.1',server_port=9999,
                             team_name=args.model,command_delay_sec=0.)
        def _process_packet(self,buffer,remote_endpoint=''):
            raw.write(struct.pack('<dI',time.time(),len(buffer)));raw.write(buffer)
            kind=struct.unpack_from('<i',buffer)[0] if len(buffer)>=4 else -1
            if kind!=2:
                raw.flush();event(kind=kind,raw=buffer.hex(),episode=self.episode)
            if kind==1:
                self.pairs.clear();self.frames=0;self.episode+=1
            if kind==2:
                p=unpack_plane_info(buffer)
                pair=self.pairs.add(p.plane_id,p.index,p)
                if pair is None:return
                if self.frames==0:event(first_pair=[asdict(x) for x in pair],episode=self.episode)
                for entry in pair:super()._handle_plane_info(entry)
                self.frames+=1
                if self.frames%600==0:
                    event(frames=self.frames,episode=self.episode,last_pair=[asdict(x) for x in pair])
                    raw.flush();print(f'frames={self.frames} episode={self.episode}',flush=True)
            else:super()._process_packet(buffer,remote_endpoint)
    event(model=model,hz=args.hz,mode='real_server_capture',scored=False)
    client=CaptureClient()
    try:client.run()
    finally:client.stop();raw.close();events.close()

if __name__=='__main__':main()
