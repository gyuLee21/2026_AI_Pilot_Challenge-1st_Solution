"""Explicit 184D -> 214D checkpoint migration; source is never modified."""
from __future__ import annotations
import argparse, copy, os, tempfile
from pathlib import Path
import torch

OLD, NEW = 184, 214

def _widen_state(state: dict) -> dict:
    out = copy.deepcopy(state); n = 0
    for key, value in list(out.items()):
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[1] == OLD and key.endswith("weight"):
            pad = torch.zeros((value.shape[0], NEW - OLD), dtype=value.dtype, device=value.device)
            out[key] = torch.cat((value, pad), dim=1); n += 1
    if n == 0: raise ValueError("no 184D input layer found")
    return out

def _widen_norm(norm):
    if not isinstance(norm, dict): raise ValueError("normalization state missing")
    out = copy.deepcopy(norm)
    for key, neutral in (("mean", 0.0), ("var", 1.0)):
        value = out.get(key)
        if not torch.is_tensor(value) or value.numel() != OLD: raise ValueError(f"normalization {key} is not 184D")
        out[key] = torch.cat((value, torch.full((NEW-OLD,), neutral, dtype=value.dtype)), dim=0)
    return out

def migrate(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve(): raise ValueError("destination must differ from source")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt["model"] = _widen_state(ckpt["model"]); ckpt["norm"] = _widen_norm(ckpt["norm"])
    for entry in ckpt.get("pool", []):
        if isinstance(entry, dict) and entry.get("model") is not None: entry["model"] = _widen_state(entry["model"])
        if isinstance(entry, dict) and entry.get("norm") is not None: entry["norm"] = _widen_norm(entry["norm"])
    ckpt["observation_migration"] = {"from": OLD, "to": NEW, "method": "zero_append_input_weights_neutral_norm_v1", "source_checkpoint": str(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent)); os.close(fd); tmp = Path(name)
    try: torch.save(ckpt, tmp); os.replace(tmp, dst)
    finally:
        if tmp.exists(): tmp.unlink()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=Path, required=True); ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args(); migrate(a.src, a.dst); print(f"migrated {a.src} -> {a.dst} ({OLD}D -> {NEW}D)")

if __name__ == "__main__": main()
