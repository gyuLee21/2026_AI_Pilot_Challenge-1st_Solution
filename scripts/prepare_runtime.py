"""Verify the archived competition SDK, or explicitly stage a local runtime.

Uses only the standard library. Never loads a DLL, starts training or downloads
files. Existing files with different hashes are always preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / 'third_party/competition_runtime'
DEFAULT_TARGET = ROOT / 'artifacts/runtimes/competition'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError(f'not a portable relative path: {relative!r}')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or str(path) != relative:
        raise ValueError(f'unsafe relative path: {relative!r}')
    full = (root / relative).resolve()
    if not full.is_relative_to(root.resolve()) or full == root.resolve():
        raise ValueError(f'path escapes root: {relative!r}')
    return full


def verify(bundle: Path) -> dict:
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('unsupported or empty runtime manifest')
    sources, targets = set(), set()
    for entry in manifest['files']:
        path = safe_path(bundle, entry['path'])
        safe_path(bundle, entry['runtime_path'])
        source_key, target_key = entry['path'].casefold(), entry['runtime_path'].casefold()
        if source_key in sources or target_key in targets:
            raise ValueError('duplicate source or destination in manifest')
        sources.add(source_key)
        targets.add(target_key)
        if not path.is_file() or path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError(f'runtime checksum mismatch: {entry["path"]}')
    return manifest


def prepare(bundle: Path, target: Path, *, install: bool = False) -> dict:
    manifest = verify(bundle)
    target = target.resolve()
    # Preflight every destination before writing anything. A second copy with a
    # different SDK version must use another directory, never an implicit force.
    pairs = []
    for entry in manifest['files']:
        source, destination = safe_path(bundle, entry['path']), safe_path(target, entry['runtime_path'])
        if destination.exists() and (not destination.is_file() or digest(destination) != entry['sha256']):
            raise FileExistsError(f'preserve different existing file: {destination}')
        for parent in destination.parents:
            if parent == target.parent:
                break
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f'destination parent is not a directory: {parent}')
        pairs.append((source, destination, entry['sha256']))
    if install:
        for source, destination, checksum in pairs:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            if digest(destination) != checksum:
                raise ValueError(f'copy verification failed: {destination}')
    return {'verified': True, 'installed': install, 'files': len(pairs),
            'target': str(target), 'dlls_loaded': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--install', action='store_true', help='Copy verified SDK files to target; default verifies only')
    args = parser.parse_args()
    try:
        result = prepare(args.bundle, args.target, install=args.install)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'runtime preparation failed: {error}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
