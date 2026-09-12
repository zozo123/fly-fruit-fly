"""Gymnasium adapter for the pinned Flybody flight task (lengths in cm)."""

import gymnasium as gym
import numpy as np


CONTROL_MODES = ("full", "frequency")


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
    """Return frame stride and realized slowdown for simulator video capture.

    A 2 ms control step rendered at 50 fps must record every control step to play
    10x slower than simulated time. Recording every tenth step would be real time.
    """
    values = (control_timestep_s, fps, slowdown)
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("control timestep, fps, and slowdown must be positive and finite")
    simulated_seconds_per_frame = 1.0 / (fps * slowdown)
    stride = max(1, int(round(simulated_seconds_per_frame / control_timestep_s)))
    realized = 1.0 / (fps * control_timestep_s * stride)
    return stride, realized


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, seed=0, render_mode=None, control_mode="full"):
        from flybody.fly_envs import flight_imitation
        from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        self.control_mode = control_mode
        self.render_mode = render_mode
        self._rng = np.random.RandomState(seed)
        self._env = flight_imitation(random_state=self._rng)
        self.dt = self._env.control_timestep()
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
        self._user_action_idx = int(self._env.task._user_idx_action)
        if control_mode == "full":
            self.action_space = gym.spaces.Box(
                self._full_action_low, self._full_action_high, dtype=np.float32,
            )
        else:
            # Learn only the WBPG frequency command in [-1, 1]. Wing residuals
            # remain exactly zero, reducing the policy action space to 1-D.
            self.action_space = gym.spaces.Box(
                np.array([-1.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32), dtype=np.float32,
            )
        obs = flatten(self._env.reset().observation)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=obs.shape, dtype=np.float32,
        )
        self._ended = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
            self.action_space.seed(seed)
        self._ended = False
        ts = self._env.reset()
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
        if self.control_mode == "full":
            return np.clip(action, self._full_action_low, self._full_action_high).copy()
        full_action = np.zeros_like(self._full_action_low)
        full_action[self._user_action_idx] = np.clip(action[0], -1.0, 1.0)
        return full_action

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset() before starting another episode")
        # Flybody adds wingbeat controls in place: always pass a fresh full action.
        ts = self._env.step(self._expand_action(action))
        terminated, truncated = end_flags(ts)
        self._ended = terminated or truncated
        return (flatten(ts.observation), float(ts.reward or 0),
                terminated, truncated, self._info(ts))

    def render(self):
        return self._env.physics.render(height=480, width=640, camera_id=1)

    def close(self):
        self._env.close()
