"""Continue this explicitly requested batch after the existing seven CPU workers.

This is a finite evaluation queue, not a recurring monitor. Stops on any error.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
import os

from tournament import save_json

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'artifacts/evaluation/final_power_tests/queue.json'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    initial=ROOT/'artifacts/evaluation/junhwa_gylee_20k_69k_cpu/report.json'
    if not initial.exists():
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000,False,args.wait_pid)
        if not handle:
            raise OSError(ctypes.get_last_error(),'Cannot wait for evaluation process')
        save_json(STATUS,dict(status='waiting_for_existing_cpu',pid=args.wait_pid,queue_pid=os.getpid()))
        try:
            while True:
                result=kernel.WaitForSingleObject(handle,30000)
                if result==0: break
                if result!=258: raise OSError('Evaluation process wait failed')
        finally:
            kernel.CloseHandle(handle)
        if not initial.exists() or not json.loads(initial.read_text()).get('complete'):
            raise RuntimeError('existing evaluation ended without complete results')
    for scenario,config in [('three_nine','power_three_nine_remaining.local.json'),
                            ('headon','power_headon_available.local.json')]:
        save_json(STATUS,dict(status='running',scenario=scenario,workers=7))
        subprocess.run([sys.executable,'-u','evaluation/native_matchups.py','--spec',
            'configs/evaluation/'+config,'--output',f'artifacts/evaluation/power_{scenario}_cpu',
            '--workers','7'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'evaluation/power_results.py'],cwd=ROOT,check=True)
    save_json(STATUS,dict(status='available_models_complete',pending='taemin.zip local download'))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save_json(STATUS,dict(status='failed',error=str(exc)))
        raise
