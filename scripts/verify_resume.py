"""Integration gate: check continuation, changed weights, and saved statistics."""
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from fly_fruit_fly.env import FlightEnv

before, after = map(Path, sys.argv[1:])
old = PPO.load(before / 'policy.zip', device='cpu')
new = PPO.load(after / 'policy.zip', device='cpu')
metadata = json.loads((after / 'training.json').read_text())
assert new.num_timesteps == old.num_timesteps + 512
assert metadata['starting_steps'] == old.num_timesteps
assert metadata['additional_steps'] == 512
assert metadata['resumed_from'] == str(before)
assert any(not np.array_equal(v.cpu().numpy(), new.policy.state_dict()[k].cpu().numpy())
           for k, v in old.policy.state_dict().items())
counts = []
for directory in (before, after):
    env = VecNormalize.load(directory / 'normalize.pkl', DummyVecEnv([FlightEnv]))
    try:
        counts.append(env.obs_rms.count)
    finally:
        env.close()
assert counts[1] > counts[0], counts
assert (after / f'rl_model_{new.num_timesteps}_steps.zip').is_file()
assert (after / f'rl_model_vecnormalize_{new.num_timesteps}_steps.pkl').is_file()
print(f'Resume verified: {old.num_timesteps} -> {new.num_timesteps} steps; weights changed; statistics continued.')
