"""Exercise the packaged EXE over the competition UDP wire protocol."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import zipfile

import numpy as np
import torch
from dogfight.unreal import protocol as wire
from dogfight.unreal.client import RemoteClientContext, PlaneSnapshot
from submission.headon_client import HeadonPolicy, TEAM
from submission.frozen_policy import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--payload', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--variant', choices=['headon', 'main'], default='headon')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    extracted = out/'standalone'
    extracted.mkdir(exist_ok=True)
    policy_class, team = HeadonPolicy, TEAM
    executable = 'ChurakDorakida_headon.exe'
    decision_hz = 60
    if args.variant == 'main':
        from submission.main_client import MainPolicy, TEAM as MAIN_TEAM
        policy_class, team = MainPolicy, MAIN_TEAM
        executable = 'ChurakDorakida.exe'
        decision_hz = 10
    with zipfile.ZipFile(args.zip) as archive:
        assert sorted(archive.namelist()) == [executable, 'config.json']
        archive.extractall(extracted)
    torch.set_num_threads(1)
    expected_policy = policy_class(load(args.payload))
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(('127.0.0.1', 9999))
    server.settimeout(60.)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PATH'] = str(Path(os.environ['SYSTEMROOT'])/'System32')
    env['CUDA_VISIBLE_DEVICES'] = ''
    log = (out/'exe_stdout.log').open('wb')
    err = (out/'exe_stderr.log').open('wb')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    began = time.perf_counter()
    proc = subprocess.Popen([str(extracted/executable)],
        cwd=out, env=env, stdout=log, stderr=err, creationflags=flags)
    results=[]
    try:
        while True:
            data, address = server.recvfrom(4096)
            if wire.unpack_message_type(data) == wire.MessageType.MT_ClientInfo:
                fields=wire.CLIENT_JOIN_INFO_STRUCT.unpack(data)
                assert fields[1].split(b'\0')[0].decode('utf-8') == team
                assert fields[2] == int(wire.AIType.ReinforcementLearning)
                break
        startup_sec=time.perf_counter()-began
        for episode in range(args.episodes):
            # Exercise both assigned aircraft IDs, swapped packet order and reset.
            own_id=episode % 2
            expected_policy.reset(None)
            server.sendto(wire.SET_PLANE_ID_STRUCT.pack(9, own_id), address)
            server.sendto(wire.INIT_STRUCT.pack(1,0,0,5000,0,0,0,300,
                                               5539,0,5000,0,0,180,300), address)
            latencies=[]
            for frame in range(args.frames):
                own = wire.PlaneInfo(frame, own_id, wire.Vector3D(frame*5,0,5000),
                    wire.Rotation3D(0,0,0), wire.Vector3D(300,0,0))
                opp = wire.PlaneInfo(frame, 1-own_id, wire.Vector3D(5539-frame*5,0,5000),
                    wire.Rotation3D(0,0,180), wire.Vector3D(300,0,0))
                context=RemoteClientContext(plane_id=own_id,frame_index=frame,
                    own_plane=PlaneSnapshot(True,own_id,frame,own),
                    enemy_plane=PlaneSnapshot(True,1-own_id,frame,opp))
                expected=expected_policy.compute_command(context)
                started=time.perf_counter()
                for p in ([own,opp] if episode==0 else [opp,own]):
                    server.sendto(wire.PLANE_INFO_STRUCT.pack(2,p.index,p.plane_id,
                        p.position.x,p.position.y,p.position.z,p.rotation.roll,p.rotation.pitch,
                        p.rotation.yaw,p.velocity.x,p.velocity.y,p.velocity.z),address)
                while True:
                    data,_=server.recvfrom(4096)
                    if wire.unpack_message_type(data)==wire.MessageType.MT_CMD:
                        break
                latency=time.perf_counter()-started
                typ,pid,index,*action=wire.CMD_STRUCT.unpack(data)
                assert pid==own_id and index==frame
                np.testing.assert_allclose(action,[expected.roll_cmd,expected.pitch_cmd,
                    expected.yaw_cmd,expected.throttle_cmd],rtol=0,atol=1e-7)
                latencies.append(latency*1000)
                time.sleep(max(0.,1/60-latency))
            assert expected_policy.provider.decisions == (args.frames-1)//(60//decision_hz)+1
            results.append(dict(episode=episode,commands=args.frames,exact_action_matches=args.frames,
                decisions=expected_policy.provider.decisions,
                first_frame_ms=latencies[0], max_frame=int(np.argmax(latencies)),
                over_budget_frames=[i for i,v in enumerate(latencies) if v>1000/60],
                frame_latencies_ms=latencies,
                latency_p50_ms=float(np.percentile(latencies,50)),
                latency_p99_ms=float(np.percentile(latencies,99)),
                latency_max_ms=max(latencies)))
        assert all(r['latency_p99_ms']<1000/60 for r in results),results
        receipt=dict(ok=True,team_name=team,decision_hz=decision_hz,startup_sec=startup_sec,episodes=results,
                     isolated_path=True,external_model_files=False,gpu_disabled=True)
        (out/'udp_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({**receipt, 'episodes':[{k:v for k,v in r.items() if k!='frame_latencies_ms'} for r in results]},ensure_ascii=True))
    finally:
        # Terminate only this test-owned packaged process and its onefile child.
        cleanup = subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
        if cleanup.returncode and proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=30)
        server.close()
        log.close()
        err.close()


if __name__=='__main__':
    main()
