"""Build the competition ZIP in an isolated CPU-PyTorch environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--payload', required=True)
    parser.add_argument('--variant', choices=['headon', 'main'], default='headon')
    args = parser.parse_args()
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError('Use the isolated CPU PyTorch build environment')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    staging = out/'build-input'
    staging.mkdir(exist_ok=True)
    is_main = args.variant == 'main'
    payload_name = 'main_policy.pt' if is_main else 'headon46k.pt'
    client_name = 'main_client.py' if is_main else 'headon_client.py'
    shutil.copy2(args.payload, staging/payload_name)
    # Package only the UDP/inference modules. Empty package initializers avoid
    # importing the training package's BT/RLlib convenience exports.
    for package in ['submission', 'claude_code', 'dogfight', 'dogfight/ai',
                    'dogfight/unreal', 'dogfight/envs', 'dogfight/sim']:
        folder = staging/package
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'__init__.py').write_text('', encoding='utf-8')
    module_paths = ['submission/headon_client.py', 'submission/decision_rate.py',
                    'submission/frozen_policy.py', 'claude_code/my_observation.py',
                    'GeoMathUtil.py', 'src/dogfight/ai/action_provider.py',
                    'src/dogfight/unreal/client.py', 'src/dogfight/unreal/protocol.py',
                    'src/dogfight/envs/observation.py', 'src/dogfight/sim/state_schema.py']
    if is_main:
        module_paths.append('submission/main_client.py')
    for relative in module_paths:
        target = relative.removeprefix('src/')
        shutil.copy2(ROOT/relative, staging/target)
    cfg = dict(server_ip='127.0.0.1', server_port=9999,
               team_name='추락도락이다' if is_main else '추락도락이다_HeadOn',
               decision_hz=10 if is_main else 60, device='cpu')
    dist = out/'files'
    dist.mkdir(exist_ok=True)
    (dist/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    name = 'ChurakDorakida' if is_main else 'ChurakDorakida_headon'
    command = [sys.executable, '-m', 'PyInstaller', str(staging/'submission'/client_name),
               '--onefile', '--console', '--noconfirm', '--name', name,
               '--distpath', str(dist), '--workpath', str(out/'work'),
               '--specpath', str(out), '--paths', str(staging),
               '--add-data', str(staging/payload_name)+';.', '--copy-metadata', 'torch']
    for module in ['cuda_fdm', 'ray', 'gymnasium', 'FighterSim', 'JSBSimWrapper',
                   'dogfight.envs.single_agent_env', 'pytest', 'tkinter', 'matplotlib',
                   'scipy', 'pandas', 'tensorboard']:
        command += ['--exclude-module', module]
    subprocess.run(command, check=True)
    exe = dist/(name+'.exe')
    result = subprocess.run([str(exe), '--self-test'], check=True, capture_output=True, text=True)
    (out/'self_test.json').write_text(result.stdout, encoding='utf-8')
    destination = out/f'추락도락이다_APTGC2026_{args.variant}.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(exe, exe.name)
        archive.write(dist/'config.json', 'config.json')
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == [exe.name, 'config.json']
        assert archive.testzip() is None
    receipt = dict(zip=str(destination), bytes=destination.stat().st_size,
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                   exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                   torch=torch.__version__, files=[exe.name, 'config.json'], config=cfg)
    (out/'build_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
