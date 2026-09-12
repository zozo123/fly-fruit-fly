"""Train, evaluate, and render a genuine physics-based flight attempt."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .env import CONTROL_MODES, video_capture_stride


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def probability(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return value


def checkpoint_config(checkpoint):
    """Read controller metadata, treating old checkpoints as full-rate/full control."""
    training = checkpoint / "training.json"
    if not training.is_file():
        return {"control_mode": "full", "action_repeat": 1}
    data = json.loads(training.read_text())
    return {
        "control_mode": data.get("control_mode", "full"),
        "action_repeat": data.get("action_repeat", 1),
    }


def train(args):
    from .env import FlightEnv
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    if args.resume:
        for filename in ("policy.zip", "normalize.pkl"):
            if not (args.resume / filename).is_file():
                raise FileNotFoundError(f"Missing resume file: {args.resume / filename}")
        previous = checkpoint_config(args.resume)
        requested = {"control_mode": args.control_mode, "action_repeat": args.action_repeat}
        if requested != previous:
            raise ValueError(
                f"Cannot resume controller config {previous!r} as {requested!r}; "
                "action space or policy timestep differs"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([
        lambda: Monitor(FlightEnv(args.seed, control_mode=args.control_mode,
                                  action_repeat=args.action_repeat))
    ])
    try:
        if args.resume:
            env = VecNormalize.load(args.resume / "normalize.pkl", env)
            env.training = True
            env.norm_reward = True
            model = PPO.load(args.resume / "policy.zip", env=env, device="cpu")
            # Resume weights, optimizer, counters and running statistics, but
            # begin a fresh seeded episode; simulator/RNG state is not saved.
            model.set_random_seed(args.seed)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10)
            model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                        n_steps=512, batch_size=64, n_epochs=5,
                        policy_kwargs={"net_arch": [64, 64]}, learning_rate=3e-4)
        starting_steps = model.num_timesteps
        if args.gamma is not None:
            model.gamma = args.gamma
            model.rollout_buffer.gamma = args.gamma
        if args.gae_lambda is not None:
            model.gae_lambda = args.gae_lambda
            model.rollout_buffer.gae_lambda = args.gae_lambda
        callback = CheckpointCallback(save_freq=args.checkpoint_every,
                                      save_path=str(args.output), save_vecnormalize=True)
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "policy.zip")
        env.save(args.output / "normalize.pkl")
        (args.output / "training.json").write_text(json.dumps({
            "seed": args.seed, "requested_steps": args.steps,
            "actual_steps": model.num_timesteps, "algorithm": "PPO",
            "connectome": False, "control_mode": args.control_mode,
            "action_repeat": args.action_repeat,
            "policy_interval_s": 0.0002 * args.action_repeat,
            "starting_steps": starting_steps,
            "additional_steps": model.num_timesteps - starting_steps,
            "resumed_from": str(args.resume) if args.resume else None,
            "checkpoint_every": args.checkpoint_every,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
        }, indent=2) + "\n")
    finally:
        env.close()


def evaluate(args):
    import imageio.v2 as imageio
    from .env import FlightEnv

    if args.checkpoint and args.expert:
        raise ValueError("Choose either --checkpoint or --expert, not both")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)

    control_mode = args.control_mode
    action_repeat = args.action_repeat
    if args.expert:
        if control_mode not in (None, "full") or action_repeat not in (None, 1):
            raise ValueError("The released expert requires full 12-D control at native rate")
        control_mode, action_repeat = "full", 1
    elif args.checkpoint:
        saved = checkpoint_config(args.checkpoint)
        if control_mode is None:
            control_mode = saved["control_mode"]
        elif control_mode != saved["control_mode"]:
            raise ValueError(
                f"Checkpoint uses {saved['control_mode']!r} control, not {control_mode!r}"
            )
        if action_repeat is None:
            action_repeat = saved["action_repeat"]
        elif action_repeat != saved["action_repeat"]:
            raise ValueError(
                f"Checkpoint uses action_repeat={saved['action_repeat']}, not {action_repeat}"
            )
    control_mode = control_mode or "full"
    action_repeat = action_repeat or 1

    env = FlightEnv(render_mode="rgb_array" if args.video else None,
                    control_mode=control_mode, action_repeat=action_repeat)
    model = normalizer = expert_policy = writer = trace = None
    episodes = []
    capture_stride = realized_slowdown = None
    try:
        if args.expert:
            from .expert import OfficialFlightPolicy
            expert_policy = OfficialFlightPolicy.from_cache(args.expert_cache)
            if env.action_space.shape != (12,):
                raise RuntimeError(
                    f"Pinned Flybody full action spec changed: {env.action_space.shape}"
                )
        elif args.checkpoint:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            model = PPO.load(args.checkpoint / "policy.zip", device="cpu")
            normalizer = VecNormalize.load(args.checkpoint / "normalize.pkl",
                                          DummyVecEnv([lambda: env]))
            normalizer.training = False
            normalizer.norm_reward = False

        trace = (args.output / "trajectory.csv").open("w", newline="")
        trace_writer = csv.DictWriter(trace, fieldnames=[
            "seed", "step", "time_s", "tracking_error_cm", "height_cm", "reward",
            "inner_control_steps", "tracking_error_sum_cm",
        ])
        trace_writer.writeheader()
        if args.video:
            capture_stride, realized_slowdown = video_capture_stride(
                env.agent_dt, fps=env.metadata["render_fps"], slowdown=args.video_slowdown
            )
            writer = imageio.get_writer(
                args.output / "flight.mp4", fps=env.metadata["render_fps"]
            )

        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            error_sum = 0.0
            policy_steps = 0
            control_steps = 0
            terminated = truncated = False
            if writer and episode == 0:
                writer.append_data(env.render())
            while not (terminated or truncated):
                if expert_policy is not None:
                    action = expert_policy.predict(env.raw_observation)
                elif model is not None:
                    action, _ = model.predict(normalizer.normalize_obs(obs.copy()),
                                              deterministic=True)
                else:
                    # Zero residuals still invoke Flybody's upstream wingbeat generator.
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("Non-finite simulation output")
                total_reward += reward
                error_sum += info["tracking_error_sum_cm"]
                control_steps += info["inner_control_steps"]
                policy_steps += 1
                trace_writer.writerow({"seed": args.seed + episode, "step": policy_steps,
                                       **info, "reward": reward})
                if (writer and episode == 0 and
                        (policy_steps % capture_stride == 0 or terminated or truncated)):
                    writer.append_data(env.render())
            episodes.append({
                "seed": args.seed + episode, "steps": policy_steps,
                "control_steps": control_steps,
                "return": total_reward, "duration_s": info["time_s"],
                "mean_tracking_error_cm": error_sum / control_steps,
                "final_height_cm": info["height_cm"],
                "failed": terminated, "completed_reference": truncated,
            })

        controller = (
            "official_flybody_expert" if expert_policy is not None
            else "ppo" if model is not None
            else "untrained_wingbeat"
        )
        report = {
            "schema_version": 5,
            "task": {"name": "flybody_straight_flight", "speed_cm_s": 20,
                     "initial_height_cm": 1, "reference_duration_s": 0.6},
            "controller": controller,
            "control_mode": control_mode,
            "action_repeat": action_repeat,
            "policy_interval_s": env.agent_dt,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "expert_source": (
                "https://janelia.figshare.com/ndownloader/files/44815195"
                if expert_policy is not None else None
            ),
            "expert_archive_sha256": (
                "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582"
                if expert_policy is not None else None
            ),
            "video_slowdown": realized_slowdown if args.video else None,
            "video_capture_stride": capture_stride if args.video else None,
            "episodes": episodes,
            "completion_rate": float(np.mean([e["completed_reference"] for e in episodes])),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "note": (
                "The official expert is an upstream released controller, not a policy trained "
                "by this repository. Completion of this synthetic task is not proof of general flight."
                if expert_policy is not None else
                "Completion of a 0.6 s synthetic reference is not proof of general flight."
            ),
        }
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if writer:
            writer.close()
        if trace:
            trace.close()
        if normalizer:
            normalizer.close()
        else:
            env.close()


def assessment(args):
    from .assessment import assess
    report = json.loads(args.metrics.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    result = assess(report, baseline)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.require_pass and not result["task_gate_passed"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Learn flight controls using PPO")
    training.add_argument("--steps", type=positive, default=1_000_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--threads", type=positive, default=2)
    training.add_argument("--resume", type=Path,
                          help="Directory containing policy.zip and matching normalize.pkl")
    training.add_argument("--control-mode", choices=CONTROL_MODES, default="full",
                          help=("full exposes all 12 upstream actions; wings exposes six wing "
                                "residuals plus WPG frequency; frequency exposes only frequency"))
    training.add_argument("--action-repeat", type=positive, default=1,
                          help="Hold each policy action for this many 0.2 ms Flybody control ticks")
    training.add_argument("--checkpoint-every", type=positive, default=10_240)
    training.add_argument("--gamma", type=probability,
                          help="Explicit discount override; otherwise preserve PPO/checkpoint default")
    training.add_argument("--gae-lambda", type=probability,
                          help="Explicit advantage-estimation horizon override")
    training.add_argument("--output", type=Path, default=Path("runs/train"))
    training.set_defaults(func=train)

    evaluation = sub.add_parser(
        "evaluate", help="Evaluate PPO, the released Flybody expert, or an untrained baseline"
    )
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--expert", action="store_true",
                            help="Use the official released Flybody flight SavedModel")
    evaluation.add_argument(
        "--expert-cache", type=Path,
        default=Path.home() / ".cache" / "fly-fruit-fly",
        help="Cache directory for the verified upstream policy archive",
    )
    evaluation.add_argument("--control-mode", choices=CONTROL_MODES,
                            help="Defaults to checkpoint metadata, or full for baseline/expert")
    evaluation.add_argument("--action-repeat", type=positive,
                            help="Defaults to checkpoint metadata, or 1 for baseline/expert")
    evaluation.add_argument("--episodes", type=positive, default=5)
    evaluation.add_argument("--seed", type=int, default=10_000)
    evaluation.add_argument("--video", action="store_true")
    evaluation.add_argument("--video-slowdown", type=positive, default=10,
                            help="Requested playback slowdown relative to simulator time")
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    evaluation.set_defaults(func=evaluate)

    gate = sub.add_parser("assess", help="Evaluate recorded metrics against fixed flight criteria")
    gate.add_argument("metrics", type=Path)
    gate.add_argument("--baseline", type=Path)
    gate.add_argument("--require-pass", action="store_true",
                      help="Exit 2 when the task gate fails; reporting alone exits 0")
    gate.set_defaults(func=assessment)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
