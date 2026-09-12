# fly-fruit-fly

**Teach a physics-based fruit fly to fly. Current status: the experiment is real; stable flight is not solved yet.**

This repository wraps [Flybody](https://github.com/TuragaLab/flybody)'s MuJoCo fruit fly and wingbeat generator with a small PPO controller. It contains executable code, trained checkpoints, raw metrics, simulator videos, failure evidence, and explicit pass/fail criteria.

It does **not** claim to contain a reconstructed fruit-fly brain or a working connectome controller.

[![Real simulator comparison: both controllers fail](media/preview.gif)](media/comparison.mp4)

**[Play the comparison MP4](media/comparison.mp4)** · [Untrained baseline](media/baseline.mp4) · [8,192-step PPO attempt](media/ppo-8192.mp4)

The video is real Flybody/MuJoCo output, approximately **10× slower than simulated time**. Orange is the simulated fly; the translucent fly is the target trajectory. The first evaluation seed is shown, not a cherry-picked success. Each panel freezes after termination and labels the failure; frozen frames are not extra flight time.

## Result so far: longer survival, worse tracking

The sustained experiment continued the original checkpoint from **8,192 to 73,728 total PPO steps**, then evaluated the learned policy and untrained wingbeat baseline on the same ten held-out initial wingbeat phases.

[Actions run 34712024080](https://github.com/zozo123/fly-fruit-fly/actions/runs/34712024080) completed successfully as an experiment, but the **flight gate failed 0/10**.

| Measurement | Untrained wingbeat | PPO, 73,728 total steps | Delta |
| --- | ---: | ---: | ---: |
| Passed 0.6 s straight-flight gate | 0 / 10 | 0 / 10 | — |
| Mean survival | 52.86 ms | 72.44 ms | +19.58 ms |
| Mean return | 95.49 | 108.70 | +13.21 |
| Mean episode tracking error¹ | 0.291 cm | 0.394 cm | **+0.104 cm worse** |

¹ Mean of each episode's mean target-position error over its own lifetime. Because episodes have different durations, this is descriptive rather than a matched-time statistical comparison.

**Interpretation:** the old controller learned behavior that survived somewhat longer and accumulated more reward, but it moved farther from the target and every episode crashed after roughly 60–83 ms—far short of the 0.6 s goal. Reward alone is therefore not a promotion criterion.

The full evidence is committed under [`results/2026-09-12-sustained/`](results/2026-09-12-sustained/): [baseline](results/2026-09-12-sustained/baseline.json), [PPO](results/2026-09-12-sustained/ppo.json), [assessment](results/2026-09-12-sustained/assessment.json), [training config](results/2026-09-12-sustained/training.json), [environment](results/2026-09-12-sustained/environment.txt), and [provenance](results/2026-09-12-sustained/provenance.json). The provenance records source commit `3eb7fc5`, Actions artifact `10303364175`, and GitHub's artifact SHA-256 digest.

Verify both the original smoke evidence and sustained result without installing MuJoCo:

```bash
python scripts/verify_results.py
```

## The next controller: wings only, slower policy

Flybody itself must still run its wingbeat generator and physics at the native **0.2 ms control interval**. PPO does not need to make a brand-new high-level decision every 0.2 ms.

The new controller keeps the inner simulator unchanged but gives PPO a cleaner interface:

```text
observations
    │
    ▼
  PPO policy ───────── every 2 ms ─────────┐
    │                                       │
    ├─ 6 wing residuals                    │ held for 10 inner ticks
    └─ 1 wingbeat-frequency command         │
                                            ▼
                              Flybody WBPG + MuJoCo
                                   every 0.2 ms
```

`--control-mode wings` exposes exactly the six Flybody wing residual channels plus the documented `user_0` wingbeat-frequency command: **7 policy actions total**. Head/abdomen/non-flight actuator channels are held at zero. `--action-repeat 10` holds each policy decision for ten native Flybody ticks, giving PPO a **2 ms policy interval** while preserving 0.2 ms wingbeat/physics integration.

This changes the learning problem from roughly **3,000 PPO decisions per 0.6 s episode to ~300**, while evaluation still accumulates tracking error and reward across every inner 0.2 ms tick. The metric is not made easier by downsampling.

At 2 ms per policy step, `gamma=0.99` has an approximate 0.2 s discount horizon. A 512-step PPO rollout spans about 1.024 s of simulated policy time and can cross episode boundaries. The 1-D `frequency` mode is also available as an ablation; it is not the primary controller.

## Run the current best experiment

Tested CI configuration: Linux, Python 3.11, CPU Torch, headless MuJoCo/EGL.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
export MUJOCO_GL=egl

# 1. Measure the matched untrained 7-D flight interface.
fly evaluate \
  --control-mode wings \
  --action-repeat 10 \
  --episodes 10 \
  --seed 30000 \
  --output runs/baseline

# 2. Train a fresh controller. Old checkpoints use a different policy interface.
fly train \
  --control-mode wings \
  --action-repeat 10 \
  --steps 8192 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --checkpoint-every 2048 \
  --output runs/candidate

# 3. Evaluate on exactly the same held-out phases and record a real video.
fly evaluate \
  --checkpoint runs/candidate \
  --episodes 10 \
  --seed 30000 \
  --video \
  --output runs/evaluation

# 4. Apply the fixed engineering gate.
fly assess \
  runs/evaluation/metrics.json \
  --baseline runs/baseline/metrics.json \
  --require-pass
```

`assess --require-pass` exits **2** when flight fails. Without that flag it records the result without turning a scientifically valid failed experiment into a broken workflow.

The same experiment is encoded in [`.github/workflows/policy-rate-flight.yml`](.github/workflows/policy-rate-flight.yml). It uploads all metrics, traces, checkpoints, normalization statistics, environment versions, and the first evaluation video whether the task passes or fails.

## What counts as "learned to fly"

The declared straight-flight gate is deliberately independent of reward:

- at least **10 held-out episodes**;
- at least **90%** must complete **≥ 0.59 s** of the 0.6 s reference;
- each passing episode must average **≤ 0.1 cm** target-position error.

This is an engineering milestone for one synthetic straight-flight task. Passing it would **not** prove takeoff, hovering, maneuverability, disturbance robustness, general flight, or biological fidelity.

Candidate/baseline comparison also requires exactly matching evaluation seeds, task configuration, controller mode, and action-repeat value. Malformed/non-finite reports, duplicate seeds, and mismatched comparisons are rejected.

## Model and environment

- Target: ~0.6 s at **20 cm/s**, starting airborne at **1 cm** center-of-mass height.
- Flybody units are centimeters, grams, and seconds.
- This project supplies a full 0.6 s synthetic reference; upstream's stock synthetic reference is much shorter.
- Flybody's wingbeat pattern generator supplies periodic wing motion. PPO learns corrections rather than inventing every wingstroke from scratch.
- The current PPO is a small two-layer **64×64 MLP** with normalized observations/rewards and CPU inference/training.
- Crashes are terminal failures. Reaching the time/reference limit is a truncation so PPO can bootstrap correctly.
- Evaluation emits `metrics.json` and `trajectory.csv`; video is evidence, not the scoring mechanism.
- Checkpoints are only reusable with their matching `normalize.pkl` and controller configuration.

The original smoke checkpoint and its evidence remain in [`results/2026-09-12-smoke/`](results/2026-09-12-smoke/). That run evaluated three phases and also failed all three; it is retained rather than replaced by the longer run.

## Replaying an existing checkpoint

```bash
fly evaluate \
  --checkpoint results/2026-09-12-smoke \
  --episodes 3 \
  --video \
  --output runs/replay
```

The legacy checkpoint is interpreted as `control_mode=full, action_repeat=1`. Resuming is allowed only when the action space and policy timestep match the saved controller metadata; changing to the 7-D wing controller requires fresh training.

## Connectome: only after embodied control works

A connectome is a later controlled experiment, not a marketing label for the PPO baseline. Once an embodied flight controller passes a meaningful task, collect observation/action pairs from it and compare a fixed sparse connectome model against both the PPO teacher and a size-matched random reservoir on unseen trajectories and disturbances.

Wiring alone does not specify synaptic signs, neuron dynamics, sensory encoding, motor decoding, or learning rules. Those assumptions must be explicit and separately tested.

## Sources

Flybody is developed by HHMI Janelia and Google DeepMind and distributed under Apache-2.0. This repository pins upstream commit [`d015e9b`](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26).

- [Whole-body physics simulation of fruit fly locomotion, Nature (2025)](https://doi.org/10.1038/s41586-025-09029-4)
- [Flybody flight environments](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/fly_envs.py)
- [Flight imitation task / wingbeat controller](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/flight_imitation.py)
- [Synthetic trajectories](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/synthetic_trajectories.py)
