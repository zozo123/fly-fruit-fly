"""Fit a nonlinear motor decoder over a fixed, pretrained connectome recurrent core."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from .assessment import assess
from .connectome import load_graph, save_graph
from .curriculum import collect_corrective_rollouts, _rollout_summary
from .flight_training import readout_dataset, validation_score
from .superfly import (collect_teacher_rollouts, evaluate, load_checkpoint,
                       save_checkpoint, set_motor_hidden)


def fit_motor(policy, x, y, epochs=100):
    """Minibatch motor fitting; graph, sensory parameters and dynamics stay fixed."""
    x, target = x[:, :-1].float(), torch.tanh(y).float()
    optimizer = torch.optim.AdamW(policy.motor.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), 256):
            idx = order[start:start+256]
            prediction = torch.tanh(policy.motor(x[idx]))
            loss = torch.mean((prediction - target[idx])**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            mse = float(torch.mean((torch.tanh(policy.motor(x))-target)**2))
        history.append(mse)
        if mse < 1e-8:
            break
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('runs/nonlinear'))
    p.add_argument('--cache', type=Path, default=Path.home()/'.cache/fly-fruit-fly')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(5678)
    graph = load_graph(args.graph)
    policy = load_checkpoint(args.checkpoint, graph)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    set_motor_hidden(policy, 128)
    save_graph(graph, args.output/'graph.npz')
    aggregate, _ = collect_teacher_rollouts(4, 1234, args.cache)
    manifest = {'algorithm':'nonlinear_motor_dagger', 'seed':5678,
                'source_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'motor_hidden':128, 'fixed':'BANC graph, sensory layer, recurrent dynamics, normalization',
                'teacher_episodes':4, 'validation_seeds':[31000,31001,31002],
                'test_seeds':list(range(80000,80010)), 'expert_actions_at_evaluation':False,
                'rounds':[]}
    best_score, best_state = None, None
    for index in range(9):
        x, y = readout_dataset(policy, aggregate)
        losses = fit_motor(policy, x, y, epochs=200 if index == 0 else 100)
        metrics = evaluate(policy, episodes=3, seed=31000,
                           output=args.output/'validation'/str(index), video=False,
                           asset_cache=args.cache)
        score = validation_score(metrics)
        manifest['rounds'].append({'round':index, 'samples':len(x), 'losses':losses,
                                   'validation_score':score})
        print(json.dumps({'round':index,'mse':losses[-1],'validation_score':score}),flush=True)
        if best_score is None or score > best_score:
            best_score, best_state = score, copy.deepcopy(policy.state_dict())
            manifest['selected_round'] = index
            save_checkpoint(policy,graph,args.output/'student.pt',manifest)
        (args.output/'training.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if best_score[0] == 3 or index == 8:
            break
        policy.load_state_dict(best_state)
        beta = max(0., .9-.15*index)
        corrective, _ = collect_corrective_rollouts(
            policy, episodes=2, seed=16000+2*index, expert_cache=args.cache, beta=beta)
        aggregate.extend(corrective)
        manifest['rounds'][-1]['corrective'] = {'beta':beta,**_rollout_summary(corrective)}
    policy.load_state_dict(best_state)
    save_checkpoint(policy,graph,args.output/'student.pt',manifest)
    result = assess(evaluate(policy,episodes=10,seed=80000,output=args.output/'evaluation',
                             video=True,asset_cache=args.cache))
    (args.output/'training.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
