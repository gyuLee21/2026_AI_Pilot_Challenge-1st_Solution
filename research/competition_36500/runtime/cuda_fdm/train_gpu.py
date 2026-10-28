# -*- coding: utf-8 -*-
"""GPU 벡터화 PPO 학습 엔트리 (GpuDogfightVecEnv + PPOGPUTrainer).

예:
  python -m cuda_fdm.train_gpu --nenv 4096 --iters 2000 --rollout 32 \
      --save runs/gpu_ppo.pt --log runs/gpu_ppo.csv

한 iteration = rollout(T) 스텝 × nac(=2·nenv) 에이전트 병렬 수집.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.finite_checks import require_finite_training_stats, record_integrity_failure
from cuda_fdm.future_aux import AUX_METRIC_KEYS

BASE_CSV_HEADER = ("iter,gstep,mean_ret,mean_len,eps,win_rate,pl,vl,ent,kl,clipfrac,"
                   "ev,sps,elapsed,pool_size,pool_perm,ema_min,ema_mean,"
                   "epochs,early_stop,optimizer_steps,actor_grad_norm,critic_grad_norm,"
                   "archive_size,league_latest,league_recent,league_core,league_challenger")


def csv_header(aux_pred):
    return BASE_CSV_HEADER + ("," + ",".join(AUX_METRIC_KEYS) if aux_pred else "")


def validate_log_schema(path, aux_pred):
    """Never append a changed training/logging protocol to an existing CSV."""
    if path and Path(path).exists() and Path(path).stat().st_size:
        with open(path, encoding="utf-8") as stream:
            actual = stream.readline().rstrip("\r\n")
        if actual != csv_header(aux_pred):
            raise ValueError("CSV schema differs (including auxiliary ON/OFF); use a separate run log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nenv", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000, help="총 main iteration(별도 exploiter iteration 제외)")
    ap.add_argument("--rollout", type=int, default=64, help="iteration 당 env step 수 T")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8,
                    help="배치(N=nenv×rollout)를 몇 조각으로 쪼갤지(=gradient step 수/epoch). "
                         "미니배치 크기가 아님. 클수록 mb 작아지고 step 많아짐")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--no-norm-obs", action="store_true")
    ap.add_argument("--num-bins", type=int, default=21, help="채널별 discrete 행동 격자 수(원본 train.py 기본=21)")
    ap.add_argument("--architecture", choices=("mlp",), default="mlp",
                    help="학습은 MLP 전용. 기존 GRU checkpoint의 변환·평가는 별도로 지원")
    ap.add_argument("--hidden", default="512,512,512",
                    help="MLP의 각 hidden layer 폭")
    ap.add_argument("--aux-pred", action=argparse.BooleanOptionalAction, default=True,
                    help="학습 전용 상대 0.5초/자신 1초 위치 residual 예측 (새 학습 기본 ON)")
    ap.add_argument("--aux-coef", type=float, default=0.1,
                    help="actor/critic 각각에 더할 auxiliary MSE 계수 (Git 기본 0.1)")
    ap.add_argument("--save-runtime", action="store_true")
    ap.add_argument("--keep-iterations", default="", help="영구 보존할 iteration 번호 CSV")
    ap.add_argument("--stop-file", default="", help="존재하면 완료된 iteration 경계에서 저장 후 중지")
    # ── iteration 스케줄 (sched-period iter 마다 단계 상승) ──
    ap.add_argument("--sched-period", type=int, default=2000,
                    help="이 iter 수마다 단계 k↑: lr·ent-coef ×= 각 decay, rollout += increment (0이면 비활성)")
    ap.add_argument("--sched-lr-decay", type=float, default=1.0 / 3.0, help="단계마다 lr 에 곱할 계수")
    ap.add_argument("--sched-ent-decay", type=float, default=1.0 / 3.0, help="단계마다 ent-coef 에 곱할 계수")
    ap.add_argument("--sched-lr-floor", type=float, default=0.0,
                    help="main schedule의 actor/critic LR 하한 (0이면 기존 동작)")
    ap.add_argument("--sched-ent-floor", type=float, default=0.0,
                    help="main schedule의 entropy loss 계수 하한; 실제 entropy 하한은 아님")
    ap.add_argument("--sched-rollout-increment", type=int, default=8, help="단계마다 rollout 에 더할 값")
    ap.add_argument("--sched-rollout-cap", type=int, default=0,
                    help="main rollout 상한(0이면 legacy unbounded schedule)")
    # ── opponent pool / gated self-play (원본과 동일 규약) ──
    ap.add_argument("--pool-evict-cap", type=int, default=4,
                    help="evictable(net) opponent snapshot 최대 수")
    ap.add_argument("--selfplay-gate-threshold", type=float, default=0.6,
                    help="evictable 최소 승률 EMA ≥ 이 값이면 현재 main 을 snapshot 추가")
    ap.add_argument("--selfplay-ema-alpha", type=float, default=0.1, help="승률 EMA 갱신율")
    ap.add_argument("--pool-sample-temp", type=float, default=0.3, help="opponent softmax 온도 τ")
    ap.add_argument("--pool-uniform-floor", type=float, default=0.5, help="샘플 균등 분배 비율 f")
    ap.add_argument("--milestone-period", type=int, default=500,
                    help="permanent snapshot(+capacity) + exploiter 학습 주기(iter, 0이면 비활성)")
    ap.add_argument("--no-opp-sample", action="store_true",
                    help="opponent 행동을 deterministic(argmax)으로")
    # ── bounded active league / cold archive ──
    ap.add_argument("--active-league", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--league-dir", default="", help="append-only cold archive directory")
    ap.add_argument("--league-active-cap", type=int, default=24)
    ap.add_argument("--league-latest-period", type=int, default=20)
    ap.add_argument("--league-recent-period", type=int, default=100)
    ap.add_argument("--league-ema-half-life-games", type=float, default=512.0)
    ap.add_argument("--league-non-latest-cap", type=float, default=0.12)
    ap.add_argument("--league-payoff-games", type=int, default=64)
    ap.add_argument("--league-payoff-refresh-period", type=int, default=500)
    ap.add_argument("--league-admission-games", type=int, default=256)
    ap.add_argument("--league-admission-score", type=float, default=0.65)
    ap.add_argument("--league-admission-lcb", type=float, default=0.60)
    ap.add_argument("--league-redteam-period", type=int, default=1000)
    # ── exploiter ──
    ap.add_argument("--exploiter-iters", type=int, default=1000,
                    help="exploiter 1회 학습 최대 iter(0 이하면 비활성)")
    ap.add_argument("--exploiter-win-target", type=float, default=0.75)
    ap.add_argument("--exploiter-alternate-altitude-hunt", action=argparse.BooleanOptionalAction, default=True,
                    help="500iter당 1개: 추락유도는 3회에 1회, 나머지는 균형/공격/수비 순환")
    ap.add_argument("--exploiter-alt-hunt-coef", type=float, default=5.0)
    ap.add_argument("--force-altitude-hunt-milestones", default="",
                    help="comma-separated main iterations at which the otherwise adaptive profile is altitude_hunt")
    ap.add_argument("--exploiter-until-iteration", type=int, default=0,
                    help="create side learners only through this main iteration (0 = no cutoff)")
    ap.add_argument("--exploiter-lr", type=float, default=1e-4)
    ap.add_argument("--exploiter-ent-coef", type=float, default=0.0001)
    ap.add_argument("--exploiter-clip-coef", type=float, default=0.2)
    ap.add_argument("--exploiter-init-iteration-first", type=int, default=500,
                    help="first iter 의 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--exploiter-init-iteration-rest", type=int, default=1000,
                    help="그 외 모든 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default=None, help="체크포인트 경로(.pt)")
    ap.add_argument("--save-every", type=int, default=100, help="N iteration 마다 저장")
    ap.add_argument("--log", type=str, default=None, help="iteration 통계 CSV 경로")
    ap.add_argument("--resume", type=str, default=None, help="이어서 학습할 .pt")
    # ── wandb (기본 켜짐; 네트워크/키 실패해도 학습은 계속) ──
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                    help="wandb 로깅 사용 (기본 켜짐)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", help="wandb 로깅 끄기")
    ap.add_argument("--wandb-project", default="AIP contest", help="wandb 프로젝트명")
    ap.add_argument("--wandb-entity", default="leeai021213-ajou-university",
                    help="W&B entity/team. 기본값은 사용자 계정의 Ajou entity")
    ap.add_argument("--wandb-expected-username", default="leeai021213",
                    help="현재 로그인 계정 검증값. 불일치 시 fail closed")
    ap.add_argument("--wandb-required", action=argparse.BooleanOptionalAction, default=True,
                    help="W&B 초기화 실패 시 학습도 중단(기본 true; --no-wandb면 미적용)")
    ap.add_argument("--wandb-run-name", default="", help="wandb run 이름 (비우면 gpu-ppo/seed)")
    ap.add_argument("--wandb-group", default="")
    args = ap.parse_args()
    if args.active_league and not args.league_dir:
        if not args.save:
            raise ValueError("--active-league requires --league-dir or --save")
        args.league_dir = str(Path(args.save).resolve().parent / "league")
    keep_iterations = {int(x) for x in args.keep_iterations.split(",") if x.strip()}
    forced_altitude_hunt_iterations = tuple(
        int(x.strip()) for x in args.force_altitude_hunt_milestones.split(",") if x.strip())

    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    if len(hidden) < 1 or any(width < 1 for width in hidden):
        raise ValueError("MLP --hidden에는 한 층 이상이 필요합니다")
    validate_log_schema(args.log, args.aux_pred)

    torch.zeros(1, device=args.device)   # CUDA 워밍업
    cfg = PPOGPUConfig(
        total_iterations=args.iters, rollout_steps=args.rollout,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_coef=args.clip,
        update_epochs=args.epochs, num_minibatches=args.minibatches,
        lr=args.lr, critic_lr=args.critic_lr, ent_coef=args.ent_coef,
        target_kl=args.target_kl, normalize_obs=not args.no_norm_obs, num_bins=args.num_bins,
        architecture=args.architecture, aux_pred=args.aux_pred, aux_coef=args.aux_coef,
        save_runtime=args.save_runtime, hidden=hidden, gru_size=0,
        sched_period=args.sched_period, sched_lr_decay=args.sched_lr_decay,
        sched_ent_decay=args.sched_ent_decay, sched_rollout_increment=args.sched_rollout_increment,
        sched_rollout_cap=args.sched_rollout_cap,
        sched_lr_floor=args.sched_lr_floor, sched_ent_floor=args.sched_ent_floor,
        pool_evict_cap=args.pool_evict_cap, selfplay_gate_threshold=args.selfplay_gate_threshold,
        selfplay_ema_alpha=args.selfplay_ema_alpha, pool_sample_temp=args.pool_sample_temp,
        pool_uniform_floor=args.pool_uniform_floor,
        milestone_period=args.milestone_period,
        opp_sample=not args.no_opp_sample,
        league_enabled=args.active_league, league_dir=args.league_dir,
        league_active_cap=args.league_active_cap,
        league_latest_period=args.league_latest_period,
        league_recent_period=args.league_recent_period,
        league_ema_half_life_games=args.league_ema_half_life_games,
        league_non_latest_cap=args.league_non_latest_cap,
        league_payoff_games=args.league_payoff_games,
        league_payoff_refresh_period=args.league_payoff_refresh_period,
        league_admission_games=args.league_admission_games,
        league_admission_score=args.league_admission_score,
        league_admission_lcb=args.league_admission_lcb,
        league_redteam_period=args.league_redteam_period,
        exploiter_iters=args.exploiter_iters, exploiter_win_target=args.exploiter_win_target,
        exploiter_alternate_altitude_hunt=args.exploiter_alternate_altitude_hunt,
        exploiter_alt_hunt_coef=args.exploiter_alt_hunt_coef,
        forced_altitude_hunt_iterations=forced_altitude_hunt_iterations,
        exploiter_until_iteration=args.exploiter_until_iteration,
        exploiter_lr=args.exploiter_lr, exploiter_ent_coef=args.exploiter_ent_coef,
        exploiter_clip_coef=args.exploiter_clip_coef,
        exploiter_init_iteration_first=args.exploiter_init_iteration_first,
        exploiter_init_iteration_rest=args.exploiter_init_iteration_rest,
        seed=args.seed, device=args.device)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    start_it = 1
    try:
        env = GpuDogfightVecEnv(args.nenv, substeps=args.substeps, seed=args.seed,
                                device=args.device)
        trainer = PPOGPUTrainer(env, cfg)
        if args.resume:
            ckpt = trainer.load(args.resume)
            start_it = ckpt.get("iteration", 1) + 1
            print(f"[resume] {args.resume} 에서 iter {start_it} 부터 재개", flush=True)
    except (FloatingPointError, ValueError) as exc:
        record_integrity_failure(Path(args.save or args.log or "runs/gpu.pt").parent,
                                 exc, "initialization_or_resume")
        raise

    # ── wandb 초기화: 소스 키 폴백 금지 + 사용자 계정 fail-closed ─────────────
    wb = None
    if args.wandb:
        try:
            import wandb
            viewer = wandb.Api().viewer
            actual_username = str(getattr(viewer, "username", "") or "")
            if args.wandb_expected_username and actual_username != args.wandb_expected_username:
                raise RuntimeError(
                    "W&B account mismatch: expected "
                    f"'{args.wandb_expected_username}', got '{actual_username or '<unknown>'}'")
            run_name = args.wandb_run_name or f"gpu-ppo/seed{args.seed}"
            # run id: '--save 파일명 stem' 같은 고정 id 는 서버에서 삭제된 run 과 충돌한다
            # (resume 으로 삭제된 id 재사용 금지). 대신 매 새 run 마다 unique id 를 생성하고
            # 체크포인트 옆 사이드카(<save>.wandbid)에 저장 → resume 시 읽어 같은 run 에 이어붙인다.
            id_path = Path(args.save).with_suffix(".wandbid") if args.save else None
            run_id = None
            attaching_existing_run = False
            if args.resume and id_path is not None and id_path.exists():
                run_id = (id_path.read_text(encoding="utf-8").strip() or None)
                attaching_existing_run = run_id is not None
            if run_id is None:
                run_id = wandb.util.generate_id()
                if id_path is not None:
                    id_path.write_text(run_id, encoding="utf-8")
            # P0 fix (2026-09-02): a resume whose checkpoint has no sidecar
            # yet (a checkpoint copied in from elsewhere, e.g. a rollback to
            # an earlier snapshot) used to still request resume="must" just
            # because --resume was passed, even though `run_id` above was
            # just freshly generated and has never existed on the server --
            # wandb correctly refused to "must"-attach to a run that was
            # never initialized. "must" is only the right fail-closed policy
            # when we actually attached to a *pre-existing* id read from the
            # sidecar (that still turns a missing/deleted/wrong id into a
            # launch failure instead of silently starting a second run with
            # overlapping iteration numbers); a brand-new id needs "allow" to
            # start its own run cleanly.
            if attaching_existing_run:
                resume_policy = "must"
            elif args.resume:
                resume_policy = "allow"
            else:
                resume_policy = "never"
            wb = wandb.init(project=args.wandb_project, entity=(args.wandb_entity or None),
                            name=run_name, id=run_id,
                            resume=resume_policy, config=vars(args),
                            group=args.wandb_group or None)
            print(f"[gpu-ppo] wandb 활성: username='{actual_username}' "
                  f"entity='{args.wandb_entity}' project='{args.wandb_project}' "
                  f"run='{run_name}' id={run_id}", flush=True)
        except Exception as e:
            if args.wandb_required:
                raise RuntimeError(f"필수 W&B 초기화 실패: {e}") from e
            print(f"[gpu-ppo] wandb 초기화 실패({e}) → wandb 없이 진행", flush=True)
            wb = None

    # ── wandb x축 정의 ──────────────────────────────────────────────────────────
    # main 지표는 전부 x축=iteration. exploiter 는 milestone 마다 별도 섹터(exploiter@500 …)
    # 를 만들고 각 섹터의 x축을 그 섹터 자체의 exp_iter(=exploiter 내부 iteration)로 둔다.
    # 이렇게 하면 explicit step 을 안 넘겨도(내부 _step 은 매 log 호출마다 단조 증가) 패널
    # x축이 섞이지 않는다 — milestone 에서 exploiter 가 수백 iter 돌아도 main step 과 충돌 없음.
    _exp_sections = set()
    if wb is not None:
        try:
            wb.define_metric("iteration")
            for _pre in ("charts", "losses", "pool", "pool_ema", "perf", "hparams", "aux"):
                wb.define_metric(f"{_pre}/*", step_metric="iteration")
            wb.define_metric("global_step", step_metric="iteration")
        except Exception as e:
            print(f"[gpu-ppo] wandb define_metric 실패({e})", flush=True)

    log_f = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "a", encoding="utf-8")
        if Path(args.log).stat().st_size == 0:
            log_f.write(csv_header(args.aux_pred) + "\n")

    print(f"[gpu-ppo] future-position auxiliary={'ON' if args.aux_pred else 'OFF'}"
          f" coef={args.aux_coef:g}; observations=184, control=10Hz", flush=True)

    def on_iter(s):
        require_finite_training_stats(s)
        wr = s.win_rate if s.win_rate == s.win_rate else float("nan")   # nan-safe
        msg = (f"it {s.iteration:5d} | ret {s.mean_return:8.3f} len {s.mean_length:6.1f} "
               f"eps {int(s.completed_episodes):5d} wr {wr:.3f} | pl {s.policy_loss:+.4f} "
               f"vl {s.value_loss:.3f} ent {s.entropy:.3f} kl {s.approx_kl:.4f} cf {s.clipfrac:.3f} "
               f"ev {s.explained_variance:+.3f} | pool {s.extra['pool_size']}"
               f"(p{s.extra['pool_perm']}) emin {s.extra['ema_min']:.3f} "
               f"| {s.steps_per_sec/1e6:.2f}M sps {s.elapsed_sec*1e3:.0f}ms")
        if args.active_league:
            msg += (f" archive {s.extra['archive_size']} "
                    f"L/R/C/X={s.extra['league_latest']}/{s.extra['league_recent']}/"
                    f"{s.extra['league_core']}/{s.extra['league_challenger']}")
        if s.extra.get("early_stop"):
            msg += f" [kl-stop @ep{s.extra['epochs']}]"
        if s.extra.get("pool_event"):
            msg += f" [{s.extra['pool_event']}]"
        if args.aux_pred:
            msg += (f" | auxMSE {s.extra['aux_actor_mse']:.5f}/{s.extra['aux_critic_mse']:.5f}"
                    f" cv={s.extra['aux_baseline_mse']:.5f}")
        print(msg, flush=True)
        if log_f is not None:
            log_f.write(f"{s.iteration},{s.global_step},{s.mean_return},{s.mean_length},"
                        f"{int(s.completed_episodes)},{s.win_rate},{s.policy_loss},{s.value_loss},"
                        f"{s.entropy},{s.approx_kl},{s.clipfrac},{s.explained_variance},"
                        f"{s.steps_per_sec},{s.elapsed_sec},{s.extra['pool_size']},"
                        f"{s.extra['pool_perm']},{s.extra['ema_min']},{s.extra['ema_mean']},"
                        f"{s.extra['epochs']},{s.extra['early_stop']},{s.extra['optimizer_steps']},"
                        f"{s.extra['actor_grad_norm']},{s.extra['critic_grad_norm']},"
                        f"{s.extra['archive_size']},{s.extra['league_latest']},"
                        f"{s.extra['league_recent']},{s.extra['league_core']},"
                        f"{s.extra['league_challenger']}"
                        + ("," + ",".join(str(s.extra[key]) for key in AUX_METRIC_KEYS)
                           if args.aux_pred else "") + "\n")
            log_f.flush()
        if wb is not None:
            try:
                logd = {
                    "charts/mean_return": s.mean_return, "charts/mean_length": s.mean_length,
                    "charts/win_rate": s.win_rate, "charts/completed_episodes": int(s.completed_episodes),
                    "charts/alt_loss_rate": s.extra["alt_loss_rate"],
                    "losses/policy_loss": s.policy_loss, "losses/value_loss": s.value_loss,
                    "losses/entropy": s.entropy, "losses/approx_kl": s.approx_kl,
                    "losses/clipfrac": s.clipfrac, "losses/explained_variance": s.explained_variance,
                    "losses/actor_grad_norm": s.extra["actor_grad_norm"],
                    "losses/critic_grad_norm": s.extra["critic_grad_norm"],
                    "perf/optimizer_steps": s.extra["optimizer_steps"],
                    "pool/size": s.extra["pool_size"], "pool/permanent": s.extra["pool_perm"],
                    "pool/ema_min": s.extra["ema_min"], "pool/ema_mean": s.extra["ema_mean"],
                    "pool/archive_size": s.extra["archive_size"],
                    "pool/latest": s.extra["league_latest"],
                    "pool/recent": s.extra["league_recent"],
                    "pool/core": s.extra["league_core"],
                    "pool/challenger": s.extra["league_challenger"],
                    "perf/steps_per_sec": s.steps_per_sec, "perf/elapsed_sec": s.elapsed_sec,
                    "hparams/lr": s.extra["lr"], "hparams/ent_coef": s.extra["ent_coef"],
                    "hparams/rollout": s.extra["rollout"],
                    "global_step": s.global_step,
                }
                if args.aux_pred:
                    logd.update({f"aux/{key.removeprefix('aux_')}": s.extra[key]
                                 for key in AUX_METRIC_KEYS})
                # opponent 별 승률 EMA. evict 슬롯은 슬롯 위치(FIFO) 기준으로 로깅해 opponent 가
                # 교체돼도 그래프 수가 안 늘어난다. permanent 는 추가 순 고정 identity 로 로깅.
                for i, ema in enumerate(s.extra.get("evict_slot_emas", [])):
                    logd[f"pool_ema/evict_slot{i}"] = ema
                for i, ema in enumerate(s.extra.get("perm_slot_emas", [])):
                    logd[f"pool_ema/perm_slot{i}"] = ema
                logd["iteration"] = s.iteration   # x축(step= 대신 step_metric 사용)
                wb.log(logd)
            except Exception as e:
                print(f"[gpu-ppo] wandb.log 실패({e})", flush=True)
        if args.save and (s.iteration % args.save_every == 0 or s.iteration in keep_iterations):
            if not all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters()):
                raise FloatingPointError(f"non-finite parameters at iteration {s.iteration}")
            trainer.save(args.save)
            if s.iteration in keep_iterations:
                dest = Path(args.save).with_name(f"iter_{s.iteration:05d}.pt")
                shutil.copy2(args.save, dest)
        # Normal training STOP is honoured only after a complete main
        # iteration (and, at milestones, after its side learner/admission), so
        # the recovery checkpoint never represents a half-applied league event.
        if args.stop_file and Path(args.stop_file).exists():
            raise KeyboardInterrupt

    def on_exploiter(ms_it, i, m):
        """milestone(ms_it)의 exploiter 학습 iteration(i) 지표를 exploiter@{ms_it} 섹터에 로깅."""
        # A small local record survives a W&B/network outage. No weights uploaded.
        if args.log:
            row = {"main_iteration": ms_it, "exploiter_iteration": i,
                   **{k: (v if not isinstance(v, float) or math.isfinite(v) else None)
                      for k, v in m.items()}}
            with Path(args.log).with_name("exploiter_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if wb is not None:
            sec = f"exploiter@{ms_it}"
            try:
                if sec not in _exp_sections:
                    wb.define_metric(f"{sec}/exp_iter")
                    wb.define_metric(f"{sec}/*", step_metric=f"{sec}/exp_iter")
                    _exp_sections.add(sec)
                d = {f"{sec}/{k}": v for k, v in m.items()}
                d[f"{sec}/exp_iter"] = i
                wb.log(d)
            except Exception as e:
                print(f"[gpu-ppo] wandb exploiter.log 실패({e})", flush=True)
        # Do not interrupt a side learner in the middle of a milestone.  The
        # completed main iteration boundary above is the resumable STOP point.

    t0 = time.time()
    failed = False
    try:
        trainer.train(on_iteration=on_iter, start_iteration=start_it,
                      on_exploiter_iter=on_exploiter)
    except KeyboardInterrupt:
        print("\n[중단] 학습 중지, 체크포인트 저장 중...", flush=True)
    except (FloatingPointError, ValueError) as exc:
        failed = True
        output_folder = Path(args.save or args.log or "runs/gpu.pt").parent
        record_integrity_failure(output_folder, exc, "training")
        raise
    except Exception:
        failed = True
        raise
    finally:
        try:
            if args.save and not failed:
                trainer.save(args.save)
                print(f"[save] {args.save}", flush=True)
        except (FloatingPointError, ValueError) as exc:
            record_integrity_failure(Path(args.save or args.log or "runs/gpu.pt").parent,
                                     exc, "checkpoint_save")
            raise
        finally:
            if log_f is not None:
                log_f.close()
            if wb is not None:
                try:
                    wb.finish()
                except Exception:
                    pass
    print(f"총 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
