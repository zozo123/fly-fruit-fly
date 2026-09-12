"""Gymnasium adapter for the pinned Flybody flight task (lengths in cm)."""

import gymnasium as gym
import numpy as np


CONTROL_MODES = ("full", "wings", "frequency")


def flatten(observation):
    return np.concatenate([
        np.asarray(observation[key]).reshape(-1) for key in sorted(observation)
    ]).astype(np.float32)


def end_flags(timestep):
    # dm_control uses discount=0 for failure and 1 for time/trajectory limits.
    ended = timestep.last()
    failed = ended and timestep.discount == 0
    return bool(failed), bool(ended and not failed)


def video_capture_stride(control_timestep_s, fps=50, slowdown=10):
    """Return frame stride and realized slowdown for simulator video capture."""
    values = (control_timestep_s, fps, slowdown)
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("control timestep, fps, and slowdown must be positive and finite")
    simulated_seconds_per_frame = 1.0 / (fps * slowdown)
    stride = max(1, int(round(simulated_seconds_per_frame / control_timestep_s)))
    realized = 1.0 / (fps * control_timestep_s * stride)
    return stride, realized


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, seed=0, render_mode=None, control_mode="full", action_repeat=1,
                 wpg_pattern_path=None):
        from flybody.fly_envs import flight_imitation
        from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        if not isinstance(action_repeat, int) or action_repeat <= 0:
            raise ValueError("action_repeat must be a positive integer")
        self.control_mode = control_mode
        self.action_repeat = action_repeat
        self.render_mode = render_mode
        self._rng = np.random.RandomState(seed)
        self._env = flight_imitation(random_state=self._rng,
                                    wpg_pattern_path=wpg_pattern_path)
        self.dt = self._env.control_timestep()
        self.agent_dt = self.dt * self.action_repeat
        # Upstream's default synthetic reference lasts only 40 ms. Supply the
        # full 0.6 s task plus lookahead, through its documented loader hook.
        qpos, qvel = constant_speed_trajectory(
            n_steps=int(round(0.6 / self.dt)) + 16,
            speed=20, init_pos=(0, 0, 1), body_rot_angle_y=-47.5,
            control_timestep=self.dt,
        )
        self._env.task._traj_generator.set_next_trajectory(qpos, qvel)
        spec = self._env.action_spec()
        self._full_action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._full_action_high = np.asarray(spec.maximum, dtype=np.float32)
        self._action_names = tuple(spec.name.split("\t")) if spec.name else ()
        self._wing_action_indices = tuple(
            int(i) for i in self._env.task._wing_inds_action
        )
        self._user_action_idx = int(self._env.task._user_idx_action)
        self._flight_action_indices = self._wing_action_indices + (self._user_action_idx,)

        if control_mode == "full":
            low, high = self._full_action_low, self._full_action_high
        elif control_mode == "wings":
            low = self._full_action_low[list(self._flight_action_indices)]
            high = self._full_action_high[list(self._flight_action_indices)]
        else:
            low = np.array([-1.0], dtype=np.float32)
            high = np.array([1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

        ts = self._env.reset()
        self._raw_observation = ts.observation
        obs = flatten(ts.observation)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=obs.shape, dtype=np.float32,
        )
        self._ended = True

    @property
    def raw_observation(self):
        """Latest native Flybody observation mapping for upstream-policy inference."""
        return self._raw_observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
            self.action_space.seed(seed)
        self._ended = False
        ts = self._env.reset()
        self._raw_observation = ts.observation
        return flatten(ts.observation), self._info(ts)

    def _info(self, ts):
        displacement = np.asarray(ts.observation["walker/ref_displacement"])
        return {
            "time_s": float(self._env.physics.time()),
            "tracking_error_cm": float(np.linalg.norm(displacement.reshape(-1, 3)[0])),
            "height_cm": float(self._env.physics.named.data.subtree_com["walker/"][2]),
        }

    def _expand_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action must have the expected shape and finite values")
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_mode == "full":
            return clipped.copy()
        full_action = np.zeros_like(self._full_action_low)
        if self.control_mode == "wings":
            full_action[list(self._flight_action_indices)] = clipped
        else:
            full_action[self._user_action_idx] = clipped[0]
        return full_action

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset() before starting another episode")
        total_reward = 0.0
        tracking_error_sum = 0.0
        ts = None
        info = None
        terminated = truncated = False
        inner_steps = 0
        for _ in range(self.action_repeat):
            # Flybody adds wingbeat controls in place, so expand a fresh full
            # action each 0.2 ms control tick even when the policy action is held.
            ts = self._env.step(self._expand_action(action))
            total_reward += float(ts.reward or 0)
            inner_steps += 1
            info = self._info(ts)
            tracking_error_sum += info["tracking_error_cm"]
            terminated, truncated = end_flags(ts)
            if terminated or truncated:
                break
        self._raw_observation = ts.observation
        self._ended = terminated or truncated
        info["inner_control_steps"] = inner_steps
        info["tracking_error_sum_cm"] = tracking_error_sum
        return (flatten(ts.observation), total_reward,
                terminated, truncated, info)

    def render(self):
        return self._env.physics.render(height=480, width=640, camera_id=1)

    def close(self):
        self._env.close()
