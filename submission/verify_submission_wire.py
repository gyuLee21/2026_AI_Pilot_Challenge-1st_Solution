"""Run a submission executable against a deterministic UDP protocol harness.

This is deliberately a wire-compatibility smoke test, not a combat simulator:
it checks the same packet ordering and lifecycle used by the local competition
server, including both assigned plane IDs and a reset between episodes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from dogfight.unreal import protocol as wire


def _send_plane(server: socket.socket, addr, p: wire.PlaneInfo) -> None:
    server.sendto(
        wire.PLANE_INFO_STRUCT.pack(
            2,
            p.index,
            p.plane_id,
            p.position.x,
            p.position.y,
            p.position.z,
            p.rotation.roll,
            p.rotation.pitch,
            p.rotation.yaw,
            p.velocity.x,
            p.velocity.y,
            p.velocity.z,
        ),
        addr,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--command", nargs=argparse.REMAINDER, default=None,
                    help="optional command vector; when set, --exe is only a label")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    exe = Path(args.exe).resolve()
    workdir = Path(args.workdir).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.command is None and not exe.is_file():
        raise FileNotFoundError(exe)
    if not workdir.is_dir():
        raise NotADirectoryError(workdir)

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 9999))
    server.settimeout(60.0)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if os.environ.get("SYSTEMROOT"):
        env["PATH"] = str(Path(os.environ["SYSTEMROOT"]) / "System32")
    env["CUDA_VISIBLE_DEVICES"] = ""
    stdout_path = out / "stdout.log"
    stderr_path = out / "stderr.log"
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    command = [str(exe)] if args.command is None else list(args.command)
    proc = subprocess.Popen(
        command, cwd=workdir, env=env, stdout=stdout, stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    result: dict = {
        "exe": str(exe),
        "team_expected": args.team,
        "frames_per_episode": args.frames,
        "episodes_requested": args.episodes,
        "ok": False,
    }
    episodes = []
    try:
        address = None
        while True:
            data, address = server.recvfrom(4096)
            if wire.unpack_message_type(data) != wire.MessageType.MT_ClientInfo:
                continue
            fields = wire.CLIENT_JOIN_INFO_STRUCT.unpack(data)
            team = fields[1].split(b"\0")[0].decode("utf-8")
            ai_type = int(fields[2])
            if team != args.team:
                raise AssertionError(f"team mismatch: {team!r} != {args.team!r}")
            result["ai_type"] = ai_type
            result["team_received"] = team
            break

        for episode in range(args.episodes):
            own_id = episode % 2
            server.sendto(wire.SET_PLANE_ID_STRUCT.pack(9, own_id), address)
            server.sendto(
                wire.INIT_STRUCT.pack(
                    1, 0, 0, 5000, 0, 0, 0, 300,
                    5539, 0, 5000, 0, 0, 180, 300,
                ),
                address,
            )
            latencies = []
            commands = []
            for frame in range(args.frames):
                own = wire.PlaneInfo(
                    frame, own_id,
                    wire.Vector3D(frame * 5, 0, 5000),
                    wire.Rotation3D(0, 0, 0),
                    wire.Vector3D(300, 0, 0),
                )
                opp = wire.PlaneInfo(
                    frame, 1 - own_id,
                    wire.Vector3D(5539 - frame * 5, 0, 5000),
                    wire.Rotation3D(0, 0, 180),
                    wire.Vector3D(300, 0, 0),
                )
                began = time.perf_counter()
                for plane in ([own, opp] if episode % 2 == 0 else [opp, own]):
                    _send_plane(server, address, plane)
                while True:
                    data, _ = server.recvfrom(4096)
                    if wire.unpack_message_type(data) == wire.MessageType.MT_CMD:
                        break
                elapsed_ms = (time.perf_counter() - began) * 1000.0
                typ, pid, index, *action = wire.CMD_STRUCT.unpack(data)
                if pid != own_id or index != frame:
                    raise AssertionError(
                        f"CMD identity mismatch: pid={pid}, index={index}, "
                        f"expected ({own_id}, {frame})"
                    )
                if not np.isfinite(np.asarray(action, dtype=np.float64)).all():
                    raise AssertionError(f"non-finite action at frame {frame}: {action}")
                latencies.append(elapsed_ms)
                commands.append(action)
                time.sleep(max(0.0, 1.0 / 60.0 - elapsed_ms / 1000.0))
            episodes.append({
                "episode": episode,
                "assigned_plane_id": own_id,
                "commands": len(commands),
                "first_frame_ms": latencies[0],
                "p50_ms": float(np.percentile(latencies, 50)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "max_ms": max(latencies),
                "over_60hz_budget": int(sum(v > (1000.0 / 60.0) for v in latencies)),
                "action_min": np.asarray(commands, dtype=np.float64).min(axis=0).tolist(),
                "action_max": np.asarray(commands, dtype=np.float64).max(axis=0).tolist(),
            })
        result["episodes"] = episodes
        result["commands_total"] = sum(e["commands"] for e in episodes)
        result["ok"] = True
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        result["process_returncode"] = proc.returncode
        server.close()
        stdout.close()
        stderr.close()
        (out / "wire_verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
