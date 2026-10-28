"""Idempotent final export. Verified bytes are reused, never regenerated.

Only a complete, checkpoint-bound staging receipt may be promoted after an
interruption. Unexpected/incomplete/corrupted files are preserved and rejected.
The caller owns the training/source integrity gates; this module owns export.
"""
from pathlib import Path


RECEIPT = "completion_receipt.json"


def complete_final_bundle(search, selected, iteration=20000):
    from claude_code.model import METADATA_FILENAME, WEIGHTS_FILENAME, load_bundle
    from cuda_fdm.finite_checks import require_finite
    from cuda_fdm.mlp_size_search import freeze_json, lock_file, read_json, sha

    final = Path(search.folder) / f"final_{iteration}"
    checkpoint, bundle = final / "checkpoint.pt", final / "bundle"
    staging, verified = final / "bundle.pending", final / "bundle_verified.json"
    expected_files = {METADATA_FILENAME, WEIGHTS_FILENAME}
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FloatingPointError("missing/unsafe final checkpoint; cannot export")
    identity = dict(passed=True, iteration=iteration, selected=selected,
                    checkpoint_sha256=sha(checkpoint))

    def validate(folder, record):
        if any(record.get(k) != v for k, v in identity.items()):
            raise FloatingPointError("bundle receipt/checkpoint/selection mismatch")
        hashes = record.get("bundle_files", {})
        if set(hashes) != expected_files:
            raise FloatingPointError("bundle receipt has unexpected file set")
        files = list(folder.iterdir()) if folder.is_dir() else []
        if {p.name for p in files} != expected_files | {RECEIPT}:
            raise FloatingPointError("bundle file set changed or is incomplete")
        if folder.is_symlink() or any(not p.is_file() or p.is_symlink() for p in files):
            raise FloatingPointError("unsafe bundle path")
        for name, digest in hashes.items():
            if sha(folder / name) != digest:
                raise FloatingPointError(f"bundle hash mismatch: {name}")
        model, metadata = load_bundle(folder, device="cpu")
        require_finite(model.state_dict(), "final_bundle.model")
        require_finite(metadata.get("obs_normalization"), "final_bundle.normalization")

    # Serializes finalization, including repeated manager launches. No GPU work.
    with lock_file(final / "bundle_finalize.lock"):
        if verified.exists():
            record = read_json(verified)
            validate(bundle, record)
            if record != read_json(bundle / RECEIPT) or staging.exists():
                raise FloatingPointError("verified bundle receipt changed or staging conflicts")
            return record  # Critically: do not call the converter or rewrite files.

        if bundle.exists():
            # Recovery after atomic rename but before writing the external marker.
            if staging.exists() or not (bundle / RECEIPT).is_file():
                raise FloatingPointError("unverified bundle lacks a complete export receipt")
            record = read_json(bundle / RECEIPT)
            validate(bundle, record)
        else:
            if staging.exists():
                if not (staging / RECEIPT).is_file():
                    raise FloatingPointError("incomplete bundle staging; preserve for review")
                record = read_json(staging / RECEIPT)
                validate(staging, record)
            else:
                staging.mkdir()
                search.command("final-bundle", ["-m", "cuda_fdm.gpu_ckpt_to_bundle",
                    "--ckpt", str(checkpoint), "--output-dir", str(staging)])
                files = list(staging.iterdir())
                if {p.name for p in files} != expected_files or any(
                        not p.is_file() or p.is_symlink() for p in files):
                    raise FloatingPointError("export produced unexpected/incomplete bundle files")
                if sha(checkpoint) != identity["checkpoint_sha256"]:
                    raise FloatingPointError("checkpoint changed during bundle export")
                model, metadata = load_bundle(staging, device="cpu")
                require_finite(model.state_dict(), "final_bundle.model")
                require_finite(metadata.get("obs_normalization"), "final_bundle.normalization")
                record = identity | {"bundle_files": {name: sha(staging / name) for name in sorted(expected_files)}}
                freeze_json(staging / RECEIPT, record)
                validate(staging, record)
            # Same filesystem; no existing destination is replaced. The receipt
            # travels atomically with the complete model and metadata.
            staging.rename(bundle)
        freeze_json(verified, record)
        return record
