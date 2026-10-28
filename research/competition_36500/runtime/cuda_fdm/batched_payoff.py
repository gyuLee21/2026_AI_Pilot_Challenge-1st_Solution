"""Independent fixed-size matchups sharing one large GPU physics batch.

Neural forwards retain the original 128-row shapes, replayed as one CUDA graph.
Per-match generators consume left sampling, right sampling and autoreset draws
in the original sequence. No changes to evaluation quotas/ICs.
"""
import math

import torch


class CapturedActions:
    """Replay unchanged kernels; warmup/capture must not advance match RNGs."""
    def __init__(self, operation, generators):
        states = [generator.get_state() for generator in generators]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        for generator in generators:
            self.graph.register_generator_state(generator)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation()
            with torch.cuda.graph(self.graph, stream=stream):
                operation()
        finally:
            torch.cuda.current_stream().wait_stream(stream)
            for generator, state in zip(generators, states):
                generator.set_state(state)

    def replay(self):
        self.graph.replay()


def sample_logits(logits, generator):
    bins = logits.shape[-1]
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    probabilities = dist.probs.reshape(-1, bins)
    return torch.multinomial(probabilities, 1, replacement=True,
                             generator=generator).reshape(-1, 4)


@torch.inference_mode()
def sample_policy(policy, observations, generator):
    net, norm, _ = policy
    if norm is not None:
        observations = norm.normalize(observations)
    logits = net.actor_logits(observations).view(-1, 4, net.num_bins)
    # Categorical(logits=...) first normalizes logits, then softmaxes. Preserve
    # those operations exactly, but omit unused log_prob/entropy computation.
    return sample_logits(logits, generator)


def batch_environment(trainer, blocks, capacity):
    from cuda_fdm.rl_env import GpuDogfightVecEnv
    from cuda_fdm.ppo_gpu import STANDARD_REWARD
    template, per_order = trainer._league_eval_env(blocks)
    key = (int(capacity), per_order)
    cache = getattr(trainer, '_batched_payoff_env', None)
    if cache is None or cache[0] != key:
        env = GpuDogfightVecEnv(2 * per_order * capacity,
            substeps=template.substeps, seed=trainer.cfg.seed, device=trainer.cfg.device,
            min_altitude_m=template.min_altitude_m,
            max_engage_time_s=template.max_engage_time_s,
            reward_cfg=template.reward_cfg, reward_mode=STANDARD_REWARD,
            alt_hunt_coef=template.alt_hunt_coef)
        trainer._batched_payoff_env = key, env
    return trainer._batched_payoff_env[1], template, per_order


@torch.inference_mode()
def evaluate_chunk(trainer, left, rights, blocks, seed_block, capacity):
    from cuda_fdm.ppo_gpu import action_to_env
    if not rights or len(rights) > capacity:
        raise ValueError('matchup count must be between one and batch capacity')
    env, template, per_order = batch_environment(trainer, blocks, capacity)
    external = any(getattr(p[0], 'obs_dim', 184)==214 for p in [left,*rights])
    if external:
        env.obr.enable_external_observations()
    lanes = 2 * per_order
    # Never create a different stratified IC distribution for the larger batch.
    template.reset_evaluation(paired=True, seed_block=seed_block)
    template._ensure_ic_pool()
    env.ic_pool, env.ic_pool_size = template.ic_pool, template.ic_pool_size
    env.sim.load_seed(template._evaluation_seed_bank.repeat(capacity, 1))
    env.obr.reset_all()
    env.obr.kernel_init_reward_state(env.sim.states)
    obs = env.obr.kernel_build_obs(env.sim.states).view(env.nenv, 2, env.OBS_SIZE)
    dev = trainer.cfg.device
    seed = (int(trainer.cfg.seed) * 1_000_003 + int(seed_block) * 97_409
            + int(blocks) * 193) % (2**31 - 1)
    generators = [torch.Generator(device=dev).manual_seed(seed) for _ in rights]
    left_slots = torch.cat((torch.zeros(per_order, dtype=torch.long, device=dev),
                            torch.ones(per_order, dtype=torch.long, device=dev)))
    left_slots_all = left_slots.repeat(capacity)
    all_rows = torch.arange(env.nenv, device=dev)
    rows = [torch.arange(i * lanes, (i + 1) * lanes, device=dev) for i in range(len(rights))]
    finished = torch.zeros(env.nenv, dtype=torch.bool, device=dev)
    finished[len(rights) * lanes:] = True
    scores = torch.full((env.nenv,), float('nan'), device=dev)
    own_crash = torch.zeros_like(finished)
    opp_crash = torch.zeros_like(finished)
    steps = [0] * len(rights)
    max_steps = math.ceil(env.max_engage_time_s / env.obr.dt) + 4
    controls = torch.zeros(env.nenv, 2, 4, device=dev)
    reset_indices = torch.zeros(env.nenv, dtype=torch.long, device=dev)
    captured = bool(getattr(trainer, '_batch_captured_actions', True))
    graph, graph_key = None, None

    def generate_actions():
        controls.zero_()
        reset_indices.zero_()
        full = env.obr.build_external_obs(env.sim.states).view(env.nenv,2,214) if external else None
        for i, right in enumerate(rights):
            if steps[i]:
                continue
            lo = full if getattr(left[0], 'obs_dim', 184)==214 else obs
            ro = full if getattr(right[0], 'obs_dim', 184)==214 else obs
            left_action = sample_policy(left, lo[rows[i], left_slots], generators[i])
            right_action = sample_policy(right, ro[rows[i], 1-left_slots], generators[i])
            controls[rows[i], left_slots] = action_to_env(left_action, trainer.cfg.num_bins, trainer.grid)
            controls[rows[i], 1-left_slots] = action_to_env(right_action, trainer.cfg.num_bins, trainer.grid)
            reset_indices[i*lanes:(i+1)*lanes] = torch.randint(
                0, env.ic_pool_size, (lanes,), device=dev, generator=generators[i])

    for step in range(1, max_steps + 1):
        if captured:
            key = (obs.data_ptr(), tuple(value == 0 for value in steps))
            if key != graph_key:
                graph = CapturedActions(generate_actions, generators)
                graph_key = key
            graph.replay()
        else:
            generate_actions()
        obs, reward, done, info = env.step(controls, reset_indices=reset_indices)
        active = ~finished
        hp, altitude = info['terminal_hp'], info['terminal_alt_m']
        finite = (info['terminal_state_finite'] & torch.isfinite(hp).all(dim=1)
                  & torch.isfinite(altitude).all(dim=1) & torch.isfinite(reward).all(dim=1))
        finite = (finite | ~active).all()
        newly = done.bool() & active
        lhp, rhp = hp[all_rows, left_slots_all], hp[all_rows, 1-left_slots_all]
        lalt, ralt = altitude[all_rows, left_slots_all], altitude[all_rows, 1-left_slots_all]
        ld = (lhp <= 0) | (lalt < env.min_altitude_m)
        rd = (rhp <= 0) | (ralt < env.min_altitude_m)
        alive = ~ld & ~rd
        win = (rd & ~ld) | (alive & (lhp > rhp + 1e-9))
        loss = (ld & ~rd) | (alive & (lhp < rhp - 1e-9))
        scores = torch.where(newly, win.float() + .5*(~win & ~loss).float(), scores)
        own_crash = torch.where(newly, lalt < env.min_altitude_m, own_crash)
        opp_crash = torch.where(newly, ralt < env.min_altitude_m, opp_crash)
        finished |= newly
        status = torch.cat((finite.reshape(1), finished.view(capacity, lanes).all(dim=1))).tolist()
        if not status[0]:
            raise FloatingPointError('non-finite batched league evaluation trajectory')
        for i in range(len(rights)):
            if status[i+1] and not steps[i]:
                steps[i] = step
        if all(steps):
            break
    if not all(steps) or not bool(torch.isfinite(scores[:len(rights)*lanes]).all()):
        raise RuntimeError('batched evaluation did not complete every required game')
    return [trainer._summarize_clean_pair(
        scores[i*lanes:(i+1)*lanes], own_crash[i*lanes:(i+1)*lanes],
        opp_crash[i*lanes:(i+1)*lanes], per_order, seed_block, seed, steps[i])
        for i in range(len(rights))]


def evaluate_row(trainer, left_bundle, identities, *, seed_block, capacity=16):
    from cuda_fdm.finite_checks import require_finite
    if trainer.cfg.architecture != 'mlp':
        raise ValueError('fixed-slot batched payoff supports the current MLP league only')
    if not 1 <= capacity <= 32 or len(set(identities)) != len(identities):
        raise ValueError('invalid capacity or duplicate matchup identity')
    saved = trainer._runtime_state()
    try:
        require_finite(left_bundle, 'batched payoff left policy')
        left = trainer._frozen_policy(left_bundle)
        result = {}
        for start in range(0, len(identities), capacity):
            ids = identities[start:start+capacity]
            bundles = [trainer.archive.load_policy(i) for i in ids]
            require_finite(bundles, 'batched payoff opponent policies')
            rights = [trainer._frozen_policy(bundle) for bundle in bundles]
            values = evaluate_chunk(trainer, left, rights,
                                    trainer.cfg.league_payoff_games, seed_block, capacity)
            result.update(zip(ids, values))
        return result
    finally:
        trainer._restore_runtime(saved)
