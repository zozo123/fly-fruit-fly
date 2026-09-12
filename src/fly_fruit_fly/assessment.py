"""Explicit engineering acceptance gate; not a biological validation claim."""
from math import isfinite
from statistics import mean


def validate(report):
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Report must contain a nonempty episode list")
    seeds = set()
    for e in episodes:
        seed = e.get("seed")
        if type(seed) is not int or seed in seeds:
            raise ValueError("Episode seeds must be unique integers")
        seeds.add(seed)
        for key in ("duration_s", "return", "mean_tracking_error_cm"):
            value = e.get(key)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f"Invalid finite metric: {key}")
        if e["duration_s"] <= 0 or e["mean_tracking_error_cm"] < 0:
            raise ValueError("Duration must be positive and error nonnegative")
        if (type(e.get("failed")) is not bool or
                type(e.get("completed_reference")) is not bool or
                e["failed"] == e["completed_reference"]):
            raise ValueError("Episode must either fail or complete the reference")
    return episodes


def controller_config(report):
    """Normalize execution metadata so legacy reports remain comparable."""
    mode = report.get("control_mode", "full")
    repeat = report.get("action_repeat", 1)
    if not isinstance(mode, str) or type(repeat) is not int or repeat <= 0:
        raise ValueError("Invalid controller configuration")
    return {"control_mode": mode, "action_repeat": repeat}


def assess(report, baseline=None):
    episodes = validate(report)
    config = controller_config(report)
    # Fixed, published task-specific criteria. Never tune these after seeing
    # results merely to make a candidate pass.
    passed = [e["completed_reference"] and e["duration_s"] >= 0.59 and
              e["mean_tracking_error_cm"] <= 0.1 for e in episodes]
    rate = mean(passed)
    result = {
        "task_gate_passed": len(episodes) >= 10 and rate >= 0.9,
        "criteria": {"minimum_episodes": 10, "minimum_pass_rate": 0.9,
                     "minimum_duration_s": 0.59, "maximum_mean_error_cm": 0.1},
        "controller_config": config,
        "episodes": len(episodes), "passing_episodes": sum(passed),
        "pass_rate": rate,
        "mean_duration_s": mean(e["duration_s"] for e in episodes),
        "mean_return": mean(e["return"] for e in episodes),
        "mean_episode_tracking_error_cm": mean(e["mean_tracking_error_cm"] for e in episodes),
        "scope": "Straight-flight task only. No takeoff, robustness, or biological validation.",
    }
    if baseline is not None:
        base = {e["seed"]: e for e in validate(baseline)}
        if set(base) != {e["seed"] for e in episodes}:
            raise ValueError("Comparison requires exactly matching evaluation seeds")
        if report.get("task") != baseline.get("task"):
            raise ValueError("Comparison requires matching task configurations")
        if config != controller_config(baseline):
            raise ValueError("Comparison requires matching controller configurations")
        result["paired_mean_delta"] = {
            key: mean(e[key] - base[e["seed"]][key] for e in episodes)
            for key in ("duration_s", "return", "mean_tracking_error_cm")
        }
        result["comparison_note"] = (
            "Descriptive paired differences, not statistical significance. "
            "Episode-mean errors cover unequal durations; use traces for matched-time comparisons."
        )
    return result
