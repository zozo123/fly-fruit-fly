"""Autonomous validation selects a connectome student; held-out tests never train it."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .assessment import assess
from .connectome import materialize_banc_subgraph
from .curriculum import collect_corrective_rollouts, _rollout_summary
from .superfly import (
    SuperFlyPolicy, _make_env, collect_teacher_rollouts, distill,
    evaluate, save_checkpoint,
)


@torch.no_grad()
def readout_dataset(policy, rollouts):
    """Replay observations through the fixed current recurrent dynamics."""
    features, targets = [], []
    for rollout in rollouts:
        state = policy.initial_state()
        for obs in rollout['observations']:
            _, _, state = policy.step(torch.as_tensor(obs, dtype=torch.float32), state)
            features.append(state[0].clone())
        native = policy.normalize_action(rollout['actions'])
        targets.append(torch.as_tensor(np.arctanh(native), dtype=torch.float64))
    if not features:
        raise ValueError('Readout fitting requires observations')
    x = torch.stack(features).double()
    return torch.cat([x, torch.ones((len(x), 1), dtype=torch.float64)], 1), torch.cat(targets)


@torch.no_grad()
def fit_readout(policy, x, y, ridge):
    """Solve mean-square pre-tanh motor prediction with L2 regularization."""
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError('ridge must be finite and positive')
    if x.ndim != 2 or y.shape != (len(x), policy.action_dim) or len(x) == 0:
        raise ValueError('Invalid readout dataset shapes')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Readout dataset must be finite')
    regularizer = torch.eye(x.shape[1], dtype=x.dtype) * (ridge * len(x))
    regularizer[-1, -1] = 0  # Do not penalize the intercept.
    solution = torch.linalg.solve(x.T @ x + regularizer, x.T @ y)
    if not torch.isfinite(solution).all():
        raise ValueError('Non-finite readout solution')
    policy.motor.weight.copy_(solution[:-1].T.float())
    policy.motor.bias.copy_(solution[-1].float())
    return float(torch.mean((torch.tanh(x @ solution) - torch.tanh(y)) ** 2))


def validation_score(metrics):
    episodes = metrics['episodes']
    passing = sum(e['completed_reference'] and e['duration_s'] >= .59 and
                  e['mean_tracking_error_cm'] <= .1 for e in episodes)
    return (passing, float(np.mean([e['duration_s'] for e in episodes])),
            -float(np.mean([e['mean_tracking_error_cm'] for e in episodes])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/flight-student'))
    p.add_argument('--cache', type=Path, default=Path.home()/'.cache/fly-fruit-fly')
    p.add_argument('--nodes', type=int, default=256)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if args.rounds < 0 or args.rounds > 30:
        p.error('rounds must be between 0 and 30')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    graph = materialize_banc_subgraph(args.cache, args.output/'graph.npz',
                                     n_nodes=args.nodes, min_synapses=5)
    env, _ = _make_env(seed=args.seed, asset_cache=args.cache)
    try:
        policy = SuperFlyPolicy(graph, int(np.prod(env.observation_space.shape)),
                                env.action_space.low, env.action_space.high)
    finally:
        env.close()
    aggregate, _ = collect_teacher_rollouts(4, args.seed, args.cache)
    losses = distill(policy, aggregate, epochs=12, learning_rate=3e-4, bptt_steps=64)
    manifest = {'algorithm': 'connectome_imitation_ridge_dagger', 'seed': args.seed,
                'graph_nodes': graph.n_nodes, 'graph_edges': graph.n_edges,
                'teacher_episodes': 4, 'initial_losses': losses,
                'normalization': 'fixed after initial teacher demonstrations',
                'validation_seeds': [30000,30001,30002],
                'held_out_test_seeds': list(range(70000,70010)),
                'expert_actions_at_evaluation': False, 'rl_steps': 0,
                'candidates': [], 'corrective_rounds': []}
    best_score = None
    best_state = None
    for round_index in range(args.rounds + 1):
        # Evaluate both the current neural fit and precise readouts. Only autonomous
        # validation selects the checkpoint; test seeds are reserved until the end.
        x, y = readout_dataset(policy, aggregate)
        candidates = [None, 1e-7, 1e-5]
        round_state = copy.deepcopy(policy.state_dict())
        round_best, round_score = None, None
        for ridge in candidates:
            policy.load_state_dict(round_state)
            loss = fit_readout(policy, x, y, ridge) if ridge else None
            label = f'round-{round_index:02d}-ridge-{ridge}'
            metrics = evaluate(policy, episodes=3, seed=30000,
                               output=args.output/'validation'/label,
                               video=False, asset_cache=args.cache)
            score = validation_score(metrics)
            manifest['candidates'].append({'name': label, 'ridge': ridge,
                                           'training_action_mse': loss,
                                           'validation_score': score})
            print(json.dumps(manifest['candidates'][-1]), flush=True)
            if round_score is None or score > round_score:
                round_score, round_best = score, copy.deepcopy(policy.state_dict())
            if best_score is None or score > best_score:
                best_score, best_state = score, copy.deepcopy(policy.state_dict())
                manifest['selected_candidate'] = label
                save_checkpoint(policy, graph, args.output/'student.pt', manifest)
            (args.output/'training.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if best_score[0] == 3 or round_index == args.rounds:
            break
        policy.load_state_dict(round_best)
        beta = max(0.0, .8 - .2 * round_index)
        corrective, _ = collect_corrective_rollouts(
            policy, episodes=2, seed=args.seed+10000+round_index*2,
            expert_cache=args.cache, beta=beta)
        aggregate.extend(corrective)
        manifest['corrective_rounds'].append({'round': round_index, 'beta': beta,
                                             **_rollout_summary(corrective)})
        print(json.dumps(manifest['corrective_rounds'][-1]), flush=True)
        # Fixed features/readout fitting is fast and avoids changing the sensory
        # coordinate system between DAgger rounds. No teacher is used at test time.
    policy.load_state_dict(best_state)
    save_checkpoint(policy, graph, args.output/'student.pt', manifest)
    metrics = evaluate(policy, episodes=10, seed=70000, output=args.output/'evaluation',
                       video=True, asset_cache=args.cache)
    result = assess(metrics)
    (args.output/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'training.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'held_out_assessment': result}), flush=True)


if __name__ == '__main__':
    main()
