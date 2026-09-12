"""Gymnasium adapter for the pinned Flybody flight task (lengths in cm)."""

import gymnasium as gym
import numpy as np


def flatten(observation):
    return np.concatenate([
        np.asarray(observation[key]).reshape(-1) for key in sorted(observation)
    ]).astype(np.float32)


def end_flags(timestep):
    # dm_control uses discount=0 for failure and 1 for time/trajectory limits.
    ended = timestep.last()
    failed = ended and timestep.discount == 0
    return bool(failed), bool(ended and not failed)


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, seed=0, render_mode=None):
        from flybody.fly_envs import flight_imitation
        from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

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
        self.action_space = gym.spaces.Box(
            np.asarray(spec.minimum, dtype=np.float32),
            np.asarray(spec.maximum, dtype=np.float32), dtype=np.float32,
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

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset() before starting another episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action must have the expected shape and finite values")
        # Flybody adds wingbeat controls in place: never pass the caller's array.
        ts = self._env.step(np.clip(action, self.action_space.low,
                                    self.action_space.high).copy())
        terminated, truncated = end_flags(ts)
        self._ended = terminated or truncated
        return (flatten(ts.observation), float(ts.reward or 0),
                terminated, truncated, self._info(ts))

    def render(self):
        return self._env.physics.render(height=480, width=640, camera_id=1)

    def close(self):
        self._env.close()
