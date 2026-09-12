import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from fly_fruit_fly.env import FlightEnv, flatten, end_flags, video_capture_stride


EXPECTED_WING_ACTIONS = {
    "wing_yaw_left", "wing_roll_left", "wing_pitch_left",
    "wing_yaw_right", "wing_roll_right", "wing_pitch_right",
}


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
    # Flybody flight control runs at 0.2 ms. Every 10th control step rendered
    # at 50 fps gives 2 ms of simulated time per 20 ms video frame: 10x slow.
    stride, slowdown = video_capture_stride(0.0002, fps=50, slowdown=10)
    assert stride == 10
    assert slowdown == pytest.approx(10.0)
    stride, slowdown = video_capture_stride(0.0002, fps=50, slowdown=1)
    assert stride == 100
    assert slowdown == pytest.approx(1.0)
    with pytest.raises(ValueError):
        video_capture_stride(0, fps=50, slowdown=10)


def test_simulator_contract_and_reproducibility():
    env = FlightEnv()
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.dt == pytest.approx(0.0002)
        # The Nature flight controller and released expert use the native
        # 12-dimensional action interface at every 0.2 ms control tick.
        assert env.action_space.shape == (12,)
        assert len(env._action_names) == 12
        assert env._action_names[env._user_action_idx] == "user_0"
        assert len(env._wing_action_indices) == 6
        assert {env._action_names[i] for i in env._wing_action_indices} == EXPECTED_WING_ACTIONS
        first, _ = env.reset(seed=42)
        assert set(env.raw_observation) == set(env._env.observation_spec())
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        np.testing.assert_array_equal(action, 0)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert info["time_s"] > 0
        assert info["inner_control_steps"] == 1
        repeated, _ = env.reset(seed=42)
        np.testing.assert_allclose(first, repeated)
        with pytest.raises(ValueError):
            env.step(np.full(env.action_space.shape, np.nan))
        # Ensure the reference really spans the full task, not the 40 ms stub.
        assert env._env.task._traj_timesteps > 2900
    finally:
        env.close()


def test_wings_controller_exposes_only_flight_channels():
    env = FlightEnv(control_mode="wings")
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.action_space.shape == (7,)
        action = np.zeros(7, dtype=np.float32)
        action[0] = 0.25
        action[-1] = -0.5
        expanded = env._expand_action(action)
        assert expanded.shape == env._full_action_low.shape
        assert expanded[env._wing_action_indices[0]] == pytest.approx(0.25)
        assert expanded[env._user_action_idx] == pytest.approx(-0.5)
        active = set(np.flatnonzero(expanded))
        assert active <= set(env._flight_action_indices)
        assert all(expanded[i] == 0 for i in range(len(expanded))
                   if i not in env._flight_action_indices)
    finally:
        env.close()


def test_action_repeat_preserves_control_rate_measurements():
    env = FlightEnv(control_mode="wings", action_repeat=10)
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.agent_dt == pytest.approx(0.002)
        env.reset(seed=5)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not (terminated and truncated)
        assert info["inner_control_steps"] == 10
        assert info["time_s"] == pytest.approx(0.002)
        assert info["tracking_error_sum_cm"] >= 0
        assert info["tracking_error_sum_cm"] >= info["tracking_error_cm"]
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
