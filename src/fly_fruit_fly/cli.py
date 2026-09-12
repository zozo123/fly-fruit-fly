"""Train, evaluate, and render a genuine physics-based flight attempt."""

import argparse
import json
from pathlib import Path

import numpy as np

from .env import FlightEnv


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    env = VecNormalize(DummyVecEnv([lambda: Monitor(FlightEnv(args.seed))]),
                       norm_obs=True, norm_reward=True, clip_obs=10)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                n_steps=512, batch_size=64, n_epochs=5,
                policy_kwargs={"net_arch": [64, 64]}, learning_rate=3e-4)
    callback = CheckpointCallback(save_freq=10_240, save_path=str(args.output),
                                  save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(args.output / "policy.zip")
        env.save(args.output / "normalize.pkl")
        (args.output / "training.json").write_text(json.dumps({
            "seed": args.seed, "requested_steps": args.steps,
            "actual_steps": model.num_timesteps, "algorithm": "PPO",
            "connectome": False,
        }, indent=2) + "\n")
    finally:
        env.close()


def evaluate(args):
    import imageio.v2 as imageio

    args.output.mkdir(parents=True, exist_ok=True)
    env = FlightEnv(render_mode="rgb_array" if args.video else None)
    model = normalizer = writer = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        model = PPO.load(args.checkpoint / "policy.zip", device="cpu")
        # The vector wrapper is used only to restore the matching statistics.
        normalizer = VecNormalize.load(args.checkpoint / "normalize.pkl",
                                      DummyVecEnv([lambda: env]))
        normalizer.training = False
        normalizer.norm_reward = False
    episodes = []
    try:
        if args.video:
            writer = imageio.get_writer(args.output / "flight.mp4", fps=50)
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward, errors, steps = 0., [], 0
            terminated = truncated = False
            if writer and episode == 0:
                writer.append_data(env.render())
            while not (terminated or truncated):
                if model is None:
                    # Zero residual still invokes upstream's wingbeat generator.
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(normalizer.normalize_obs(obs.copy()),
                                              deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("Non-finite simulation output")
                total_reward += reward
                errors.append(info["tracking_error_cm"])
                steps += 1
                # 10x slow motion: 10 physics-control steps per video frame.
                if writer and episode == 0 and (steps % 10 == 0 or terminated or truncated):
                    writer.append_data(env.render())
            episodes.append({
                "seed": args.seed + episode, "steps": steps,
                "return": total_reward, "duration_s": info["time_s"],
                "mean_tracking_error_cm": float(np.mean(errors)),
                "final_height_cm": info["height_cm"],
                "failed": terminated, "completed_reference": truncated,
            })
        report = {
            "controller": "ppo" if model else "untrained_wingbeat",
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "video_slowdown": 10 if args.video else None,
            "episodes": episodes,
            "completion_rate": float(np.mean([e["completed_reference"] for e in episodes])),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "note": "Completion of a 0.6 s synthetic reference is not proof of general flight.",
        }
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if writer:
            writer.close()
        if normalizer:
            normalizer.close()
        else:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Learn wingbeat corrections using PPO")
    training.add_argument("--steps", type=positive, default=1_000_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--threads", type=positive, default=2)
    training.add_argument("--output", type=Path, default=Path("runs/train"))
    training.set_defaults(func=train)
    evaluation = sub.add_parser("evaluate", help="Evaluate PPO or the untrained baseline")
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--episodes", type=positive, default=5)
    evaluation.add_argument("--seed", type=int, default=10_000)
    evaluation.add_argument("--video", action="store_true")
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    evaluation.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
