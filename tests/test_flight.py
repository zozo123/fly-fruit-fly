import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from fly_fruit_fly.env import FlightEnv, flatten, end_flags, video_capture_stride


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


def test_video_capture_timing():
    stride, slowdown = video_capture_stride(0.002, fps=50, slowdown=10)
    assert stride == 1
    assert slowdown == pytest.approx(10.0)
    stride, slowdown = video_capture_stride(0.002, fps=50, slowdown=1)
    assert stride == 10
    assert slowdown == pytest.approx(1.0)
    with pytest.raises(ValueError):
        video_capture_stride(0, fps=50, slowdown=10)


def test_simulator_contract_and_reproducibility():
    env = FlightEnv()
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.dt == pytest.approx(0.002)
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
        assert env._env.task._traj_timesteps > 290
    finally:
        env.close()


def test_frequency_only_controller_expands_to_user_action():
    env = FlightEnv(control_mode="frequency")
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.action_space.shape == (1,)
        expanded = env._expand_action(np.array([0.25], dtype=np.float32))
        assert expanded.shape == env._full_action_low.shape
        assert expanded[env._user_action_idx] == pytest.approx(0.25)
        assert np.count_nonzero(expanded) == 1
        env.reset(seed=7)
        obs, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    finally:
        env.close()
