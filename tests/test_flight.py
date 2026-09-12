import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from fly_fruit_fly.env import FlightEnv, flatten, end_flags


def test_flatten_order():
    np.testing.assert_array_equal(flatten({"z": [3], "a": [[1, 2]]}), [1, 2, 3])


@pytest.mark.parametrize("last,discount,expected", [
    (False, 1, (False, False)), (True, 0, (True, False)),
    (True, 1, (False, True)),
])
def test_termination_semantics(last, discount, expected):
    class Timestep:
        def last(self):
            return last
    ts = Timestep()
    ts.discount = discount
    assert end_flags(ts) == expected


def test_simulator_contract_and_reproducibility():
    env = FlightEnv()
    try:
        check_env(env, warn=True, skip_render_check=True)
        first, _ = env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        np.testing.assert_array_equal(action, 0)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert info["time_s"] > 0
        repeated, _ = env.reset(seed=42)
        np.testing.assert_allclose(first, repeated)
        with pytest.raises(ValueError):
            env.step(np.full(env.action_space.shape, np.nan))
        # Ensure the reference really spans the full task, not the 40 ms stub.
        assert env._env.task._traj_timesteps > 2900
    finally:
        env.close()
