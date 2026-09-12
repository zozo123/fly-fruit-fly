"""Check recorded evidence and recompute published numbers using stdlib only."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results/2026-09-12-smoke"
SUSTAINED = ROOT / "results/2026-09-12-sustained"


def episode_summary(report):
    episodes = report["episodes"]
    return {
        "mean_duration_s": mean(e["duration_s"] for e in episodes),
        "mean_return": mean(e["return"] for e in episodes),
        "mean_episode_tracking_error_cm": mean(
            e["mean_tracking_error_cm"] for e in episodes
        ),
    }


def verify_smoke():
    provenance = json.loads((SMOKE / "provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"Hash mismatch: {name}")
    for name in ["baseline", "ppo"]:
        report = json.loads((SMOKE / f"{name}.json").read_text())
        episodes = report["episodes"]
        assert [e["seed"] for e in episodes] == [10000, 10001, 10002]
        assert len(episodes) == 3 and all(e["failed"] for e in episodes)
        assert not any(e["completed_reference"] for e in episodes)
        assert abs(report["mean_return"] - mean(e["return"] for e in episodes)) < 1e-8
        summary = episode_summary(report)
        print(name, json.dumps({
            "mean_duration_ms": 1000 * summary["mean_duration_s"],
            "mean_return": summary["mean_return"],
            "mean_episode_tracking_error_cm": summary["mean_episode_tracking_error_cm"],
            "completed": 0,
            "failed": 3,
        }))
    training = json.loads((SMOKE / "training.json").read_text())
    assert training["actual_steps"] == 8192 and training["seed"] == 0
    print(f"Verified {len(provenance['sha256'])} smoke file hashes and recorded metrics.")


def verify_sustained():
    baseline = json.loads((SUSTAINED / "baseline.json").read_text())
    candidate = json.loads((SUSTAINED / "ppo.json").read_text())
    assessment = json.loads((SUSTAINED / "assessment.json").read_text())
    training = json.loads((SUSTAINED / "training.json").read_text())
    provenance = json.loads((SUSTAINED / "provenance.json").read_text())

    expected_seeds = list(range(20000, 20010))
    for report in (baseline, candidate):
        episodes = report["episodes"]
        assert [e["seed"] for e in episodes] == expected_seeds
        assert all(e["failed"] and not e["completed_reference"] for e in episodes)
        assert report["completion_rate"] == 0
        assert abs(report["mean_return"] - episode_summary(report)["mean_return"]) < 1e-8

    base = episode_summary(baseline)
    cand = episode_summary(candidate)
    assert assessment["task_gate_passed"] is False
    assert assessment["passing_episodes"] == 0 and assessment["pass_rate"] == 0
    for key, value in cand.items():
        assert abs(assessment[key] - value) < 1e-12

    paired_summary_keys = {
        "duration_s": "mean_duration_s",
        "return": "mean_return",
        "mean_tracking_error_cm": "mean_episode_tracking_error_cm",
    }
    for episode_key, value in assessment["paired_mean_delta"].items():
        summary_key = paired_summary_keys[episode_key]
        assert abs(value - (cand[summary_key] - base[summary_key])) < 1e-12

    assert training == {
        "seed": 0,
        "requested_steps": 65536,
        "actual_steps": 73728,
        "algorithm": "PPO",
        "connectome": False,
        "starting_steps": 8192,
        "additional_steps": 65536,
        "resumed_from": "results/2026-09-12-smoke",
        "checkpoint_every": 10240,
        "gamma": 0.999,
        "gae_lambda": 0.99,
    }
    assert provenance["workflow_run_id"] == 34712024080
    assert provenance["head_sha"] == "3eb7fc501895f59e53d853b9f594ad16b1baf17c"
    assert provenance["artifact_digest"] == (
        "sha256:2066e789d5328e964d97dd24e84ef57a0cd47af7416404ba9e0b7c3a68bd3e8a"
    )
    env_hash = hashlib.sha256((SUSTAINED / "environment.txt").read_bytes()).hexdigest()
    assert env_hash == provenance["artifact_file_sha256"]["environment.txt"]
    print("sustained", json.dumps({
        "baseline": base,
        "candidate": cand,
        "paired_delta": assessment["paired_mean_delta"],
        "passing_episodes": assessment["passing_episodes"],
    }))


def main():
    verify_smoke()
    verify_sustained()


if __name__ == "__main__":
    main()
