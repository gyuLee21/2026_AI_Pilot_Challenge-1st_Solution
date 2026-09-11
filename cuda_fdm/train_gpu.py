# -*- coding: utf-8 -*-
"""GPU 벡터화 PPO 학습 엔트리 (GpuDogfightVecEnv + PPOGPUTrainer).

예:
  python -m cuda_fdm.train_gpu --nenv 4096 --iters 2000 --rollout 32 \
      --save runs/gpu_ppo.pt --log runs/gpu_ppo.csv

한 iteration = rollout(T) 스텝 × nac(=2·nenv) 에이전트 병렬 수집.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
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
from cuda_fdm.training_control import (RunLease, resolve_deadline,
                                      retarget_resume_state)

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


def _main(leases):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduled-legacy-import", default="",
                    help="Optional one-shot external challenger batch manifest")
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
    ap.add_argument("--training-stop-config", default="",
                    help="Opt-in frozen 3-9 suite / plateau / 5000-iter polish config JSON")
    ap.add_argument("--evaluate-stop-baseline-only", action="store_true",
                    help="Run/resume stop baseline evaluation, save, and exit without a PPO iteration")
    ap.add_argument("--accept-target-change", action="store_true",
                    help="resume에서 총 iteration/ETA 목표 변경을 명시적으로 승인")
    deadline_group = ap.add_mutually_exclusive_group()
    deadline_group.add_argument("--deadline", default=None,
                                help="timezone 포함 ISO 시각, 예: 2026-09-13T16:00:00+09:00")
    deadline_group.add_argument("--clear-deadline", action="store_true",
                                help="checkpoint에 저장된 마감 설정을 명시적으로 해제")
    ap.add_argument("--deadline-reserve-seconds", type=float, default=None)
    ap.add_argument("--deadline-milestone-floor-seconds", type=float, default=None)
    ap.add_argument("--deadline-safety-factor", type=float, default=None)
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
                    help="schedule rollout 상한(0이면 legacy 무제한 증가)")
    ap.add_argument("--main-finish-iteration", type=int, default=0)
    ap.add_argument("--main-finish-rollout", type=int, default=96)
    ap.add_argument("--main-finish-actor-lr", type=float, default=3e-5)
    ap.add_argument("--main-finish-critic-lr", type=float, default=5e-5)
    ap.add_argument("--main-finish-entropy", type=float, default=5e-5)
    # ── opponent pool / gated self-play (원본과 동일 규약) ──
    ap.add_argument("--pool-evict-cap", type=int, default=4,
                    help="evictable(net) opponent snapshot 최대 수")
    ap.add_argument("--selfplay-gate-threshold", type=float, default=0.6,
                    help="evictable 최소 승률 EMA ≥ 이 값이면 현재 main 을 snapshot 추가")
    ap.add_argument("--selfplay-ema-alpha", type=float, default=0.1, help="승률 EMA 갱신율(legacy 비리그 전용; --active-league 시 무효)")
    ap.add_argument("--pool-sample-temp", type=float, default=0.3, help="opponent softmax 온도 τ")
    ap.add_argument("--pool-uniform-floor", type=float, default=0.5, help="샘플 균등 분배 비율 f")
    ap.add_argument("--milestone-period", type=int, default=500,
                    help="permanent snapshot(+capacity) + exploiter 학습 주기(iter, 0이면 비활성)")
    ap.add_argument("--exploiter-period", type=int, default=0,
                    help="side learner cadence; 0 follows milestone-period, otherwise a positive multiple")
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
    ap.add_argument("--league-altitude-redteam-threshold", type=float, default=0.10)
    ap.add_argument("--league-altitude-hunter", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable altitude-hunt side learners, including guaranteed league sentinels")
    # ── isolated 100K+ control plane (legacy default is byte-for-byte behaviour) ──
    ap.add_argument("--vnext-mode", choices=("staged",), default="staged")
    ap.add_argument("--vnext-stage", type=int, default=1)
    ap.add_argument("--vnext-state-dir", default="")
    ap.add_argument("--vnext-strategic-budget", type=int, default=64)
    # 2026-09-03: PayoffGraphConfig's own default is 4, which was sized for a
    # run that inherits 20K's already-populated payoff graph. From an empty
    # archive the mandatory admission queries (priority 100+) take the budget
    # first, so low-priority *solver* edges -- the ones that make a policy
    # solver-eligible and therefore roster-eligible -- accrue only from
    # whatever is left over. Raising the budget lets solver edges build
    # alongside admission work instead of competing with it, so the strategic
    # core fills in the opening milestones rather than trickling.
    ap.add_argument("--vnext-query-budget", type=int, default=12,
                    help="payoff queries per milestone (vNext PayoffGraphConfig)")
    # ── exploiter ──
    ap.add_argument("--exploiter-iters", type=int, default=1000,
                    help="exploiter 1회 학습 최대 iter(0 이하면 비활성)")
    ap.add_argument("--exploiter-win-target", type=float, default=0.80)
    ap.add_argument("--exploiter-alternate-altitude-hunt", action=argparse.BooleanOptionalAction, default=True,
                    help="legacy archive-None 전용(3회에 1회 추락유도 순환). 리그 모드에서는 무효: "
                    "프로파일은 bandit + ALTITUDE_SENTINEL_PERIOD/OFFSET 강제 헌터가 결정")
    ap.add_argument("--exploiter-alt-hunt-coef", type=float, default=5.0)
    ap.add_argument("--exploiter-lr", type=float, default=1e-4)
    ap.add_argument("--exploiter-ent-coef", type=float, default=0.0001)
    ap.add_argument("--exploiter-clip-coef", type=float, default=0.2)
    ap.add_argument("--exploiter-init-iteration-first", type=int, default=500,
                    help="first iter 의 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--exploiter-init-iteration-rest", type=int, default=1000,
                    help="그 외 모든 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--substeps", type=int, default=6)
    # 2026-09-03 (rule change): two separate submissions, one per initial
    # condition. This locks BOTH the training reset distribution and the
    # league evaluation bank to one scenario, so admission/solver decisions
    # are never measured on the other scenario's games. "mixed" reproduces
    # the previous 3-9:head-on = 3:1 blend.
    ap.add_argument("--scenario", type=str, default="mixed",
                    choices=("three_nine", "headon", "mixed"),
                    help="초기조건 고정: 3-9 전용 / head-on 전용 / 기존 혼합")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headon-distance-m", type=float, default=5539.0,
                    help="Head-on initial separation in metres (server distance)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default=None, help="체크포인트 경로(.pt)")
    ap.add_argument("--save-every", type=int, default=100, help="N iteration 마다 저장")
    ap.add_argument("--log", type=str, default=None, help="iteration 통계 CSV 경로")
    ap.add_argument("--resume", type=str, default=None, help="이어서 학습할 .pt")
    ap.add_argument(
        "--force-historical-reactivation-id", type=int, default=None,
        help=("resume clean boundary에서 명시한 archive policy를 historical "
              "probationary로 재등록한 후 기존 payoff/Nash/forced-Challenger "
              "activation transaction으로 한 번 투입"))
    ap.add_argument("--damage-scale", type=float, default=10.0)
    ap.add_argument("--headon-damage-schedule", action="store_true",
                    help="Headon only: damage scale 10 before iteration 20000, then 2")
    ap.add_argument("--altitude-settlement-scale", type=float, default=10.0,
                    help="remaining-HP altitude settlement, independent of damage shaping")
    ap.add_argument("--timeout-draw-reward", type=float, default=-4.0)
    ap.add_argument("--altitude-terminal-mode", choices=("remaining_hp", "result", "result_remaining_hp"), default="remaining_hp")
    ap.add_argument("--altitude-win-reward", type=float, default=5.0)
    ap.add_argument("--altitude-loss-reward", type=float, default=-5.0)
    ap.add_argument("--accept-objective-change", action="store_true",
                    help="explicitly accept a reward/gamma retune across resume")
    ap.add_argument("--accept-schedule-change", action="store_true", default=False,
                    help="resume 시 스케줄 감쇠율/floor 변경을 의도된 것으로 승인 "
                         "(base LR/entropy·period·rollout/shaping ladder 변경은 여전히 거부)")
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
    ap.add_argument("--wandb-run-id", default="", help="coordinator가 승인한 고정 W&B run id")
    ap.add_argument("--wandb-group", default="")
    args = ap.parse_args()
    if args.headon_damage_schedule and (args.scenario != "headon" or args.damage_scale != 10.0):
        raise ValueError("headon damage schedule requires headon and initial damage-scale 10")
    stop_config = (json.loads(Path(args.training_stop_config).read_text(encoding="utf-8"))
                   if args.training_stop_config else None)
    legacy_plan = None
    if args.scheduled_legacy_import:
        from cuda_fdm.scheduled_legacy_import import load_plan
        legacy_plan = load_plan(args.scheduled_legacy_import)
        if args.scenario != "headon" or not args.save or not args.active_league:
            raise ValueError("legacy batch requires headon, active league, and checkpoint output")
    if args.evaluate_stop_baseline_only and stop_config is None:
        raise ValueError("--evaluate-stop-baseline-only requires --training-stop-config")
    if args.iters < 1:
        raise ValueError("--iters must be a positive total main iteration target")
    if args.deadline is not None or not args.resume:
        initial_deadline = resolve_deadline(
            deadline=args.deadline, clear=args.clear_deadline,
            reserve=args.deadline_reserve_seconds,
            milestone_floor=args.deadline_milestone_floor_seconds,
            safety_factor=args.deadline_safety_factor)
        if (initial_deadline is not None and not args.resume
                and initial_deadline.epoch <= time.time() + initial_deadline.reserve_seconds):
            raise ValueError("cannot start fresh inside the deadline reserve window")
    if args.active_league and not args.league_dir:
        if not args.save:
            raise ValueError("--active-league requires --league-dir or --save")
        args.league_dir = str(Path(args.save).resolve().parent / "league")
    keep_iterations = {int(x) for x in args.keep_iterations.split(",") if x.strip()}
    if args.save:
        leases.enter_context(RunLease(Path(args.save).parent, resume=bool(args.resume)))

    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    if len(hidden) < 1 or any(width < 1 for width in hidden):
        raise ValueError("MLP --hidden에는 한 층 이상이 필요합니다")
    validate_log_schema(args.log, args.aux_pred)

    torch.zeros(1, device=args.device)   # CUDA 워밍업
    cfg = PPOGPUConfig(
        headon_damage_schedule=args.headon_damage_schedule,
        total_iterations=args.iters, rollout_steps=args.rollout,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_coef=args.clip,
        update_epochs=args.epochs, num_minibatches=args.minibatches,
        lr=args.lr, critic_lr=args.critic_lr, ent_coef=args.ent_coef,
        target_kl=args.target_kl, normalize_obs=not args.no_norm_obs, num_bins=args.num_bins,
        architecture=args.architecture, aux_pred=args.aux_pred, aux_coef=args.aux_coef,
        save_runtime=args.save_runtime, hidden=hidden, gru_size=0,
        sched_period=args.sched_period, sched_lr_decay=args.sched_lr_decay,
        sched_ent_decay=args.sched_ent_decay, sched_rollout_increment=args.sched_rollout_increment,
        sched_lr_floor=args.sched_lr_floor, sched_ent_floor=args.sched_ent_floor,
        sched_rollout_cap=args.sched_rollout_cap,
        main_finish_iteration=args.main_finish_iteration,
        main_finish_rollout=args.main_finish_rollout,
        main_finish_actor_lr=args.main_finish_actor_lr,
        main_finish_critic_lr=args.main_finish_critic_lr,
        main_finish_entropy=args.main_finish_entropy,
        pool_evict_cap=args.pool_evict_cap, selfplay_gate_threshold=args.selfplay_gate_threshold,
        selfplay_ema_alpha=args.selfplay_ema_alpha, pool_sample_temp=args.pool_sample_temp,
        pool_uniform_floor=args.pool_uniform_floor,
        milestone_period=args.milestone_period,
        exploiter_period=args.exploiter_period,
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
        league_altitude_redteam_threshold=args.league_altitude_redteam_threshold,
        league_altitude_hunter=args.league_altitude_hunter,
        exploiter_iters=args.exploiter_iters, exploiter_win_target=args.exploiter_win_target,
        exploiter_alternate_altitude_hunt=args.exploiter_alternate_altitude_hunt,
        exploiter_alt_hunt_coef=args.exploiter_alt_hunt_coef,
        exploiter_lr=args.exploiter_lr, exploiter_ent_coef=args.exploiter_ent_coef,
        exploiter_clip_coef=args.exploiter_clip_coef,
        exploiter_init_iteration_first=args.exploiter_init_iteration_first,
        exploiter_init_iteration_rest=args.exploiter_init_iteration_rest,
        seed=args.seed, device=args.device)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    start_it = 1
    resume_checkpoint = None
    try:
        env = GpuDogfightVecEnv(args.nenv, substeps=args.substeps, seed=args.seed,
                                device=args.device, scenario=args.scenario,
                                headon_distance_m=args.headon_distance_m)
        env.reward_cfg.update(damage_scale=args.damage_scale,
                              altitude_settlement_scale=args.altitude_settlement_scale,
                              altitude_terminal_mode=args.altitude_terminal_mode,
                              altitude_win_reward=args.altitude_win_reward,
                              altitude_loss_reward=args.altitude_loss_reward,
                              timeout_draw_reward=args.timeout_draw_reward)
        _scenario = getattr(env, "scenario", "?")
        _prob = getattr(env, "scenario_b_prob", float("nan"))
        try:
            _prob_text = f"{float(_prob):g}"
        except (TypeError, ValueError):
            _prob_text = repr(_prob)
        print(f"[gpu-ppo] initial-condition scenario={_scenario} "
              f"(head-on probability {_prob_text}); "
              f"headon_distance_m={args.headon_distance_m}; "
              "league evaluation bank is locked to the same scenario", flush=True)
        trainer = PPOGPUTrainer(env, cfg)
        print(f"[reward] training objective={trainer._training_objective()}", flush=True)
        if args.resume:
            resume_checkpoint = trainer.load(
                args.resume, allow_schedule_change=args.accept_schedule_change,
                allow_objective_change=args.accept_objective_change)
            start_it = resume_checkpoint.get("iteration", 1) + 1
            print(f"[resume] {args.resume} 에서 iter {start_it} 부터 재개", flush=True)
            # 2026-09-04 FIX (C1): drop uncheckpointed tail rows so resume
            # never appends duplicate iter rows (checkpoint is every 100,
            # metrics flush every iter). Atomic rewrite.
            if args.log:
                _log_path = Path(args.log)
                if _log_path.is_file() and _log_path.stat().st_size > 0:
                    _ckpt_iter = int(resume_checkpoint.get("iteration", 0))
                    _lines = _log_path.read_text(encoding="utf-8").splitlines()
                    if _lines:
                        _header, _rows = _lines[0], _lines[1:]
                        _kept = [_header]
                        for _row in _rows:
                            _cell = _row.split(",", 1)[0].strip()
                            try:
                                if int(float(_cell)) <= _ckpt_iter:
                                    _kept.append(_row)
                            except (TypeError, ValueError):
                                _kept.append(_row)
                        _dropped = len(_rows) - (len(_kept) - 1)
                        if _dropped:
                            _tmp = _log_path.with_suffix(".csv.tmp")
                            _tmp.write_text("\n".join(_kept) + "\n", encoding="utf-8")
                            os.replace(_tmp, _log_path)
                            print(f"[resume] metrics.csv tail {_dropped}행 제거"
                                  f" (checkpoint iter {_ckpt_iter} 이후 미커밋분)", flush=True)
    except (FloatingPointError, ValueError) as exc:
        record_integrity_failure(Path(args.save or args.log or "runs/gpu.pt").parent,
                                 exc, "initialization_or_resume")
        raise

    # 2026-09-03: this isolated 100K package always trains from scratch
    # (fresh model, no migrated 20K checkpoint) and always runs staged/live
    # vNext control from iteration 1 -- the shadow-calibration and
    # migration-coordinator machinery this file used to gate on (a written
    # launch receipt approving the live adapter) has been removed along
    # with the coordinator itself. source_iteration=0 means "no pre-run
    # archive baseline", the correct value for a from-scratch run (see
    # league_vnext/contracts.py's VNextConfig.validate()).
    if not args.active_league or not args.vnext_state_dir:
        raise ValueError("vNext requires --active-league and --vnext-state-dir")
    from cuda_fdm.league_vnext.contracts import (
        PayoffGraphConfig, StrategicIndexConfig, VNextConfig, VNextMode, VNextStage)
    from cuda_fdm.league_vnext.shadow import VNextShadowController
    from cuda_fdm.league_vnext.live_adapter import LIVE_ADAPTER_PROTOCOL, VNextMilestoneAdapter
    index_config = replace(StrategicIndexConfig(),
                           soft_budget=int(args.vnext_strategic_budget))
    payoff_config = replace(PayoffGraphConfig(),
                            query_budget_per_milestone=int(args.vnext_query_budget))
    if args.scenario == "headon":
        from .headon_distance_change import distance_payoff_config
        payoff_config = distance_payoff_config(payoff_config,args.headon_distance_m)
    vnext_config = VNextConfig(
        mode=VNextMode(args.vnext_mode), stage=VNextStage(args.vnext_stage),
        source_iteration=0, target_iteration=int(args.iters), strategic_index=index_config,
        payoff_graph=payoff_config)
    vnext_state = (resume_checkpoint or {}).get("vnext_control")
    external_vnext_state = Path(args.vnext_state_dir) / "shadow_state.json"
    if vnext_state is None and external_vnext_state.is_file():
        vnext_state = json.loads(external_vnext_state.read_text(encoding="utf-8"))
    if vnext_state is not None and resume_checkpoint is None:
        raise ValueError("fresh training cannot adopt existing vNext state; use a new run root")
    vnext_state, target_change = retarget_resume_state(
        vnext_state, target=args.iters, completed=start_it-1,
        saved_main_target=(resume_checkpoint or {}).get('cfg', {}).get('total_iterations'),
        allow_change=args.accept_target_change)
    deadline_guard = resolve_deadline(
        saved=(resume_checkpoint or {}).get('training_deadline'), deadline=args.deadline,
        clear=args.clear_deadline, reserve=args.deadline_reserve_seconds,
        milestone_floor=args.deadline_milestone_floor_seconds,
        safety_factor=args.deadline_safety_factor)
    trainer.training_deadline = None if deadline_guard is None else deadline_guard.state_dict()
    if vnext_state is None and start_it != vnext_config.source_iteration + 1:
        raise RuntimeError(
            "first vNext start must be a fresh (non-resumed) launch at iteration 1")
    vnext = VNextShadowController(
        args.vnext_state_dir, config=vnext_config, state=vnext_state)
    if target_change is not None:
        vnext.log.append("target_iteration_changed", target_change, iteration=start_it-1)
        print(f"[target] {target_change}", flush=True)
    # The written "launch receipt" this used to require was a coordinator
    # process's sign-off artifact; launching this CLI with --vnext-mode
    # staged directly *is* that approval now, so this is synthesized here
    # rather than read from a file.
    trainer.vnext_milestone_adapter = VNextMilestoneAdapter(
        vnext, release_receipt={"protocol": LIVE_ADAPTER_PROTOCOL, "approved": True,
                                "stage": int(vnext_config.stage)})
    trainer.vnext_control_state = vnext.state_dict()
    print(f"[gpu-ppo] vNext {args.vnext_mode} control enabled at stage "
          f"{args.vnext_stage}; live_adapter="
          f"{trainer.vnext_milestone_adapter is not None}", flush=True)

    from .headon_distance_change import finish_pending_rebaseline
    finish_pending_rebaseline(trainer,args.save)

    from .warm_start import finish_pending
    finish_pending(trainer,args.save)

    from .legacy_league import finish_pending as finish_legacy_league
    finish_legacy_league(trainer,args.save,(resume_checkpoint or {}).get('legacy_league_transition'))

    # Explicit operator-directed safety recovery. This is an opt-in,
    # resume-only operation: it does not weaken the normal UCB gate or turn a
    # fixed-suite diagnostic into admission evidence. Membership is published
    # only by the ordinary atomic payoff -> Nash -> forced-Challenger path.
    if args.force_historical_reactivation_id is not None:
        if resume_checkpoint is None or not args.save:
            raise ValueError("forced historical reactivation requires --resume and --save")
        identity = int(args.force_historical_reactivation_id)
        record = trainer.archive.records.get(identity)
        if record is None:
            raise ValueError(f"historical archive {identity} does not exist")
        if record.get("safety_status", "valid") != "valid":
            raise ValueError(f"historical archive {identity} is not safety-valid")
        current_id = trainer._last_milestone_archive_id
        if current_id is None:
            raise RuntimeError("forced historical reactivation needs current milestone archive")
        already_active = any(
            entry.get("archive_id") is not None
            and int(entry["archive_id"]) == identity
            for entry in trainer.pool.active_entries())
        already_solved = identity in set(map(int, vnext.last_solver_ids))
        if not (already_active and already_solved):
            if record.get("admission_status") != "archive_only":
                raise ValueError(
                    f"historical archive {identity} is not archive_only: "
                    f"{record.get('admission_status')!r}")
            metrics = record.setdefault("metrics", {})
            record["admitted"] = True
            record["admission_status"] = "probationary"
            metrics["historical_counter_reactivated"] = True
            metrics["historical_counter_reactivated_at_iteration"] = int(trainer.iteration)
            metrics["historical_counter_reactivation_count"] = int(metrics.get(
                "historical_counter_reactivation_count", 0)) + 1
            metrics["manual_safety_reactivation"] = True
            metrics["manual_safety_reactivation_reason"] = (
                "operator_confirmed_fixed_suite_altitude_regression")
            activated = trainer.vnext_milestone_adapter.activate_post_side_candidate(
                trainer, archive_id=identity, iteration=int(trainer.iteration),
                current_id=int(current_id), recovery=True)
            if not activated:
                raise RuntimeError(
                    f"historical archive {identity} was not activated; inspect pending state")
            trainer.vnext_control_state = vnext.state_dict()
            trainer.save(args.save)
            print(f"[resume] historical archive {identity}를 iteration "
                  f"{trainer.iteration} challenger로 재투입하고 저장", flush=True)
        else:
            print(f"[resume] historical archive {identity}는 이미 active solver member", flush=True)

    # 2026-09-05: releases before post-side activation support could archive a
    # direct-entry altitude sentinel as admitted/probationary without placing
    # it in either the active pool or the complete solver game. Repair that
    # exact stranded state before the first resumed rollout, then checkpoint
    # the repaired league atomically at the same committed Main iteration.
    if (resume_checkpoint is not None
            and trainer._last_milestone_archive_id is not None):
        recovered_ids = trainer.vnext_milestone_adapter.reconcile_stranded_post_side_candidate(
            trainer, iteration=int(trainer.iteration),
            current_id=int(trainer._last_milestone_archive_id))
        # Refresh derived observational state at the same committed boundary.
        # save() preserves Main/optimizer/runtime RNG and synchronizes resident
        # online statistics and final admission receipts; no payoff re-query.
        if not args.save:
            raise RuntimeError("resume reconciliation requires --save")
        trainer.vnext_control_state = vnext.state_dict()
        trainer.save(args.save)
        print(f"[resume] pool online/membership metadata synchronized at clean "
              f"iteration {trainer.iteration}", flush=True)
        if recovered_ids:
            print(f"[resume] stranded post-side archive {list(recovered_ids)}를 "
                  f"iteration {trainer.iteration} challenger로 복구하고 저장", flush=True)

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
            run_id = args.wandb_run_id.strip() or None
            if args.resume and id_path is not None and id_path.exists():
                saved_run_id = id_path.read_text(encoding="utf-8").strip() or None
                if run_id is not None and saved_run_id != run_id:
                    raise RuntimeError("approved W&B run id differs from checkpoint sidecar")
                run_id = saved_run_id
            # 2026-09-04 FIX (C4): resume with a missing sidecar previously forked
            # a silent new W&B lineage. Fail loudly instead.
            if args.resume and id_path is not None and not id_path.exists() and run_id is None:
                raise RuntimeError(
                    "W&B sidecar missing on resume "
                    f"({id_path}); restore <save>.wandbid from backup or pass "
                    "--wandb-run-id explicitly to fork intentionally")
            if run_id is None:
                run_id = wandb.util.generate_id()
            if id_path is not None:
                id_path.write_text(run_id, encoding="utf-8")
            wb = wandb.init(project=args.wandb_project, entity=(args.wandb_entity or None),
                            name=run_name, id=run_id,
                            resume="allow", config=vars(args), group=args.wandb_group or None)
            wb.config.update({"gamma": args.gamma, "damage_scale": args.damage_scale,
                              "headon_distance_m": args.headon_distance_m,
                              "headon_distance_transition": getattr(trainer,"headon_distance_transition",None),
                              "legacy_league_transition": getattr(trainer,"legacy_league_transition",None),
                              "timeout_draw_reward": args.timeout_draw_reward,
                              "altitude_settlement_scale": args.altitude_settlement_scale,
                              "altitude_terminal_mode": args.altitude_terminal_mode,
                              "altitude_win_reward": args.altitude_win_reward,
                              "altitude_loss_reward": args.altitude_loss_reward,
                              "training_objective": trainer._training_objective()},
                             allow_val_change=True)
            wb.summary["training_objective_history"] = getattr(trainer, "training_objective_history", [])
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
          f" coef={args.aux_coef:g}; observations=214, control=10Hz", flush=True)

    # Resume may already be exactly at the approved import boundary.
    # Complete the existing transaction before collecting the next rollout.
    if legacy_plan is not None and resume_checkpoint is not None:
        from cuda_fdm.scheduled_legacy_import import apply_if_due
        apply_if_due(trainer, legacy_plan, args.save)

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
                    "charts/own_alt_event_rate_v2": s.extra["own_alt_event_rate_v2"],
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
                if vnext is not None:
                    logd.update({f"vnext_health/{key}": value
                                 for key, value in s.extra.items()
                                 if key.startswith((
                                     "entropy_head", "normalized_entropy_head",
                                     "effective_actions_head", "support_retention",
                                     "support_fraction", "probe_js_drift",
                                     "top2_probability",
                                     "max_probability_head", "ratio_q", "clipfrac_",
                                     "actor_logit_rms"))})
                # opponent 별 승률 EMA. evict 슬롯은 슬롯 위치(FIFO) 기준으로 로깅해 opponent 가
                # 교체돼도 그래프 수가 안 늘어난다. permanent 는 추가 순 고정 identity 로 로깅.
                for i, ema in enumerate(s.extra.get("evict_slot_emas", [])):
                    logd[f"pool_ema/evict_slot{i}"] = ema
                for i, ema in enumerate(s.extra.get("perm_slot_emas", [])):
                    logd[f"pool_ema/perm_slot{i}"] = ema
                logd["iteration"] = s.iteration   # x축(step= 대신 step_metric 사용)
                logd["reward/damage_scale"] = float(env.reward_cfg["damage_scale"])
                wb.log(logd)
            except Exception as e:
                print(f"[gpu-ppo] wandb.log 실패({e})", flush=True)
        if vnext is not None:
            health_metrics = {
                "approx_kl": s.approx_kl,
                "clipfrac": s.clipfrac,
                "actor_grad_norm": s.extra["actor_grad_norm"],
                "critic_grad_norm": s.extra["critic_grad_norm"],
                "entropy": s.entropy,
                "explained_variance": s.explained_variance,
                # The current collector is synchronous on-policy. Future
                # asynchronous collectors must replace these with measured
                # values rather than retaining optimistic constants.
                "fresh_fraction": 1.0,
                "policy_lag": 0,
                "optimizer_steps": s.extra["optimizer_steps"],
                "rollout": s.extra["rollout"],
                "elapsed_sec": s.elapsed_sec,
                "environment_transitions": s.extra["environment_transitions"],
                "completed_games": s.extra["completed_games"],
                "evaluation_elapsed_sec": s.extra["evaluation_elapsed_sec"],
                "side_elapsed_sec": s.extra["side_elapsed_sec"],
                "opponent_exposure": s.extra.get("opponent_exposure", []),
            }
            health_metrics.update({key: value for key, value in s.extra.items()
                                   if key.startswith((
                                       "entropy_head", "normalized_entropy_head",
                                       "effective_actions_head", "support_retention",
                                       "support_fraction", "probe_js_drift",
                                       "top2_probability",
                                       "max_probability_head", "ratio_q", "clipfrac_",
                                       "actor_logit_rms"))})
            vnext.observe_iteration(s.iteration, health_metrics)
            # Staged mode's milestone bookkeeping (which also calls
            # controller.observe_milestone internally, with the live
            # trainer's actual active/candidate pools rather than this
            # shadow-only sentinel/heldout approximation) happens through
            # trainer.vnext_milestone_adapter.on_milestone() in ppo_gpu.py's
            # main loop -- nothing further is needed here.
            trainer.vnext_control_state = vnext.state_dict()
            if s.iteration % 100 == 0 or s.iteration == args.iters:
                budget_report = vnext.budget.report(
                    iteration=s.iteration, milestone_period=args.milestone_period)
                print(f"[budget] target={args.iters} "
                      f"remaining={budget_report['remaining_main_iterations']} "
                      f"milestones={budget_report['remaining_milestones']} "
                      f"eta_seconds={budget_report['rolling_eta_seconds']}", flush=True)
                if wb is not None:
                    try:
                        wb.log({"iteration": s.iteration, "perf/target_iteration": args.iters,
                                "perf/eta_seconds": budget_report['rolling_eta_seconds']})
                    except Exception as exc:
                        print(f"[budget] W&B ETA log failed: {exc}", flush=True)
        if legacy_plan is not None:
            from cuda_fdm.scheduled_legacy_import import apply_if_due
            apply_if_due(trainer, legacy_plan, args.save)
        if args.save and (s.iteration % args.save_every == 0 or s.iteration in keep_iterations):
            if not all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters()):
                raise FloatingPointError(f"non-finite parameters at iteration {s.iteration}")
            trainer.save(args.save)
            if s.iteration in keep_iterations:
                dest = Path(args.save).with_name(f"iter_{s.iteration:05d}.pt")
                # 2026-09-04 FIX (C3): atomic keep-copy (kill mid-copy no
                # longer leaves a partial iter_*.pt).
                _tmp_keep = dest.with_suffix(".pt.tmp")
                shutil.copy2(args.save, _tmp_keep)
                os.replace(_tmp_keep, dest)
        # Normal training STOP is honoured only after a complete main
        # iteration (and, at milestones, after its side learner/admission), so
        # the recovery checkpoint never represents a half-applied league event.
        if args.stop_file and Path(args.stop_file).exists():
            raise KeyboardInterrupt
        if stopper is not None and stopper.due():
            stopper.tick()

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
    stopper = None
    if getattr(trainer, "training_stop_state", None) and stop_config is None:
        raise ValueError("resume requires the saved --training-stop-config; cannot silently disable")
    if stop_config is not None:
        from cuda_fdm.training_stop import TrainingStopController
        stopper = TrainingStopController(trainer, stop_config, args.save,
                                         log_callback=wb.log if wb is not None else None)
        if wb is not None:
            wb.config.update({"iters": args.iters, "training_stop": stop_config},
                             allow_val_change=True)
    failed = False
    def before_iteration(iteration):
        if args.evaluate_stop_baseline_only:
            return False
        if args.stop_file and Path(args.stop_file).exists():
            return False
        if stopper is not None and not stopper.before_iteration(iteration):
            return False
        if deadline_guard is None:
            return True
        receipt = deadline_guard.check(
            next_iteration=iteration, target_iteration=args.iters,
            milestone_period=args.milestone_period, budget=vnext.budget)
        if (not receipt['allow'] or receipt['is_milestone']
                or iteration == start_it or iteration % 100 == 0):
            vnext.log.append("deadline_gate", receipt, iteration=iteration-1)
            trainer.vnext_control_state = vnext.state_dict()
            print(f"[deadline] {receipt}", flush=True)
        return receipt['allow']

    try:
        if stopper is not None and stopper.due():
            # Includes restart of a partially cached evaluation, before rollout.
            stopper.tick()
        if args.resume and args.accept_objective_change and args.save:
            # Publish the approved objective and unchanged clean learner state
            # before collecting any transitions under the new reward/gamma.
            trainer.save(args.save)
            print(f"[save] objective migration committed at iteration {start_it - 1}", flush=True)
        trainer.train(on_iteration=on_iter, start_iteration=start_it,
                      on_exploiter_iter=on_exploiter,
                      before_iteration=before_iteration if deadline_guard is not None or stopper is not None else None)
    except KeyboardInterrupt:
        if trainer.checkpoint_safe:
            print("\n[중단] clean iteration 경계에서 체크포인트 저장 중...", flush=True)
        else:
            print("\n[중단] iteration 도중 중지되어 기존 recovery checkpoint를 보존합니다.",
                  flush=True)
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
            if args.save and not failed and trainer.checkpoint_safe:
                trainer.save(args.save)
                print(f"[save] {args.save}", flush=True)
            elif args.save and not failed:
                print("[save] in-flight iteration은 저장하지 않았습니다; 기존 checkpoint 유지",
                      flush=True)
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


def main():
    with ExitStack() as leases:
        return _main(leases)


if __name__ == "__main__":
    main()
