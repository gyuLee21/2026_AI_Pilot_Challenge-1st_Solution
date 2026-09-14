"""Extract frozen policies, preserve normalization, verify, and package for sharing."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from junhwa_policy import JunhwaPolicy
from cuda_fdm.ppo_gpu import build_actor_critic, RunningNorm, action_to_env
from cuda_fdm.future_aux import inference_state_dict

OUT=ROOT.parent/'outputs/2026-09-11_team_models'
TEMPLATE=ROOT.parent/'outputs/2026-09-11_gylee_winners/2026-09-11_gylee_three_nine_iter69000'
BASE=ROOT/'artifacts/models'
SOURCES=[
 ('gylee_20k',BASE/'rl/common/legacy20k_obs214/original20k_20000.pt',20000,'common'),
 ('gylee_69k',BASE/'rl/3-9/completed_70k/three_nine_iter_69000.pt',69000,'three_nine'),
 ('gylee_46k_headon',BASE/'rl/headon/completed_20k_resume/headon_iter_46000.pt',46000,'headon'),
]+[(p.stem,p,None,'common') for p in sorted((BASE/'external/inbox/junhwa/common').glob('*.pt'))]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    torch.set_num_threads(1)
    template_cfg=torch.load(SOURCES[1][1],map_location='cpu',weights_only=False,mmap=True)['cfg']
    OUT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name,source,iteration,scenario in SOURCES:
        target=OUT/name
        target.mkdir(exist_ok=True)
        foreign=name.startswith('junhwa_')
        c=torch.load(source,map_location='cpu',weights_only=foreign,mmap=True)
        cfg=c.get('cfg',template_cfg)
        iteration=iteration or c.get('iteration')
        weights=c['model']
        actor={k:v.detach().clone() for k,v in weights.items()
               if k.startswith('actor_') and not k.startswith('actor_aux_head.')}
        if not foreign:
            last=max(int(k.split('.')[1]) for k in actor if k.endswith('.weight'))
            actor={(('actor_logits.'+k.split('.')[-1]) if int(k.split('.')[1])==last
                    else 'actor_body.'+'.'.join(k.split('.')[1:])):v for k,v in actor.items()}
        norm={k:torch.as_tensor(v).clone() for k,v in c['norm'].items()}
        if not foreign:
            norm={k:v.float() for k,v in norm.items()}  # Runtime RunningNorm stores FP32.
        compact=dict(format='frozen_actor_with_norm_v1',iteration=iteration,
            cfg={k:cfg[k] for k in ('hidden','activation','num_bins','gru_last','gru_all') if k in cfg},
            model=actor,norm=norm)
        compact['cfg'].setdefault('activation','tanh')
        compact['cfg'].setdefault('num_bins',21)
        torch.save(compact,target/'model.pt')
        deployed=JunhwaPolicy(torch.load(target/'model.pt',weights_only=True)).eval()
        if foreign:
            reference=JunhwaPolicy(c).eval()
            reference_norm=None
        else:
            reference=build_actor_critic(obs_dim=214,act_dim=4,num_bins=cfg['num_bins'],
                architecture='mlp',hidden=tuple(cfg['hidden']),gru_size=0,
                encoder_depth=cfg.get('encoder_depth',2),activation=cfg['activation']).eval()
            reference.load_state_dict(inference_state_dict(weights,enabled=bool(cfg.get('aux_pred',False))),strict=True)
            reference_norm=RunningNorm(214,'cpu')
            reference_norm.load_state_dict(c['norm'])
        torch.manual_seed(260911)
        sequences=[torch.randn(256,214) for _ in range(4)]
        max_error=0.
        for device in ('cpu','cuda'):
            deployed=deployed.to(device)
            reference=reference.to(device)
            if reference_norm is not None:
                reference_norm=RunningNorm(214,device)
                reference_norm.load_state_dict(c['norm'])
            a=deployed.initial_state(256,device)
            b=reference.initial_state(256,device)
            with torch.inference_mode():
                for t,x in enumerate(sequences):
                    x=x.to(device)
                    starts=torch.zeros(256,dtype=torch.bool,device=device)
                    starts[:64]=True
                    if t==0: starts[:]=True
                    got,a=deployed.act(x,a,starts)
                    want,b=reference.act(reference_norm.normalize(x) if reference_norm else x,b,starts,sample=False)
                    assert torch.equal(got,want),(name,device,t,'actions differ')
                    if a is not None:
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    if not foreign:
                        n=(x-deployed.mean)/torch.sqrt(deployed.var+1e-8)
                        n=n.clamp(-10,10).float()
                        logits=deployed.actor_logits(deployed.actor_body(n))
                        ref_logits=reference.actor_logits(reference_norm.normalize(x))
                        max_error=max(max_error,(logits-ref_logits).abs().max().item())
                        torch.testing.assert_close(logits,ref_logits,rtol=0,atol=0)
        for folder in ('claude_code','src'):
            shutil.copytree(TEMPLATE/folder,target/folder,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for filename in ('my_observation.py','observation_contract.py'):
            shutil.copy2(ROOT/'claude_code'/filename,target/'claude_code'/filename)
        shutil.copy2(TEMPLATE/'GeoMathUtil.py',target/'GeoMathUtil.py')
        shutil.copy2(TEMPLATE/'requirements.txt',target/'requirements.txt')
        shutil.copy2(ROOT/'evaluation/share_inference.py',target/'inference.py')
        shutil.copy2(ROOT/'evaluation/junhwa_policy.py',target/'frozen_actor.py')
        shutil.copy2(ROOT/'evaluation/SHARING_README_KO.md',target/'README.md')
        metadata=dict(name=name,iteration=iteration,scenario=scenario,source_name=source.name,
            source_sha256=sha(source),model_sha256=sha(target/'model.pt'),observation_size=214,
            observation_contract='cuda_claude164r_command_poststep_v1',
            observation_contract_evidence='shared-code owner declaration' if foreign else 'training code',
            policy_hz=10,action_repeat=6,action_mode='argmax',channels=['roll','pitch','rudder','throttle'],
            normalization='stored mean/var, sqrt(var+1e-8), clip [-10,10], frozen',
            cfg=compact['cfg'],recurrent=bool(compact['cfg'].get('gru_last',False)),
            actor_layout={k:list(v.shape) for k,v in actor.items()},
            omitted=['optimizer','critic','auxiliary head','pool','training RNG'],
            original20k_conversion=('legacy184_to_obs214_zero_acceleration_columns_v1: old indices 0:164 -> 0:164; 164:184 -> 194:214; new acceleration columns have zero weights' if name=='gylee_20k' else None))
        save(target/'metadata.json',metadata)
        verification=dict(source_actor_tensors_preserved=True,normalization_preserved=True,
            reference='existing Junhwa evaluator' if foreign else 'original gylee actor and RunningNorm',
            devices=['cpu','cuda'],batch_size=256,sequence_steps=4,episode_reset_test=True,
            argmax_exact_match=True,hidden_state_exact_match=True,max_gylee_logit_error=max_error)
        save(target/'verification.json',verification)
        archive=OUT/(name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(target.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':
                    z.write(f,f.relative_to(target))
        outputs.append(dict(name=name,path=str(archive),size=archive.stat().st_size,sha256=sha(archive)))
        print(name,'PASS',archive.stat().st_size,flush=True)
    save(OUT/'packages.json',outputs)

if __name__=='__main__': main()
