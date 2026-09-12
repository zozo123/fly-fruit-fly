# fly-fruit-fly

Teach a simulated fruit fly to follow a straight flight path, using
[Flybody](https://github.com/TuragaLab/flybody)'s actual MuJoCo body and wingbeat
generator. A small PPO policy learns corrections to the existing wing controls.

**First milestone:** a reproducible training run, saved controller, evaluation
metrics, and video. This is a flight-learning baseline, not a reconstructed fly
brain. No connectome data or pretrained controller is bundled. Short smoke runs
validate the pipeline; they do not establish successful learned flight.

## Run

Use Python 3.11 on Linux for the tested CI configuration. Python 3.12 is also
allowed but not covered by CI. From a checkout of this repository:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
export MUJOCO_GL=egl

# Measure the existing wingbeat generator with no learned corrections.
fly evaluate --episodes 5 --video --output runs/baseline

# Train the small controller. This is a starting budget, not a success guarantee.
fly train --steps 1000000 --output runs/train

# Reload policy AND observation statistics; use the same evaluation seeds.
fly evaluate --checkpoint runs/train --episodes 5 --video --output runs/evaluation
```

`python -m pip install -e '.[dev]'` works instead of `uv pip install` in an
activated Python environment. For CPU-only Torch, install its CPU wheel first as
shown in the workflow. On headless Ubuntu, install `libegl1-mesa`,
`libgl1-mesa-dri`, and `ffmpeg`. On a desktop, omit the EGL setting if the normal
MuJoCo renderer works. Training itself does not capture video.

## No local setup: GitHub Actions

Open [Actions → Flight baseline](../../actions/workflows/flight.yml).
Each push runs simulator tests, measures the untrained controller, trains for
8,192 steps, reloads the saved model, and renders an evaluation. This is a smoke
budget, not enough to assume convergence. For a longer experiment, choose
**Run workflow** and enter the training budget. The job has a 60-minute cap;
checkpoints and normalization statistics are saved every 10,240 steps and
uploaded even if a later step fails. Runs consume GitHub Actions minutes.

Download the run's `flight-baseline-…` artifact to get:

| File | Purpose |
| --- | --- |
| `baseline/metrics.json` | Untrained wingbeat reference score |
| `baseline/flight.mp4` | First untrained episode, including any crash |
| `train/policy.zip` | Trained PPO model |
| `train/normalize.pkl` | Matching observation normalization; keep with model |
| `train/training.json` | Seed and actual training steps |
| `evaluation/metrics.json` | Held-out episode scores, durations, errors, failures |
| `evaluation/flight.mp4` | First evaluation episode, including any crash |
| `environment.txt` | Exact installed dependency versions |

Videos show the real simulator at **10× slow motion**. The translucent ghost is
the target trajectory, not the learned fly. Files are written under `runs/`
locally; workflow artifacts expire after 14 days, so download runs you want to keep.
Only load trusted model/normalization files.

## What the experiment means

- Target: approximately 0.6 seconds at **20 cm/s**, **1 cm** initial CoM height.
  The fly starts airborne at the target velocity; takeoff is not trained.
- Flybody's units are centimeters, grams, and seconds. Its stock synthetic
  reference lasts only 40 ms; this project supplies a full-length reference.
- Physics and aerodynamic forces come from Flybody/MuJoCo. The wingbeat
  generator supplies a periodic baseline; PPO learns residual actions.
- Rewards are Flybody's existing position/orientation tracking and leg terms.
  Crashes terminate episodes; reference/time limits truncate them so PPO can
  bootstrap correctly.
- Observations are flattened in stable key order and normalized. The tiny
  two-layer, 64-unit policy runs on CPU. This PPO path is our baseline, not the
  upstream paper's DMPO training reproduction.
- Compare completion rate, mean return, and per-episode tracking error against
  the untrained baseline on the same held-out seeds. Seeds vary initial wingbeat
  phase; they do not represent diverse environments. Longer survival alone
  does not establish accurate flight. Repeat training across seeds before
  claiming a reliable improvement.

## Connectome: next controlled experiment

Once the flight baseline works, collect observation/action pairs from its
controller. Fit an input adapter and readout around a fixed, sparse connectome
network, first by imitation and then optionally by reinforcement learning.
Compare against the PPO teacher and a size-matched random reservoir on unseen
trajectories and disturbances. Measure whether the real wiring helps.

Wiring counts alone do not specify synaptic signs, neuron dynamics, sensory
encoding, or motor decoding. Keep those modeling assumptions explicit. Do not
download gigabytes of connectome data until the embodied baseline is measured.

## Sources and attribution

Flybody is developed by HHMI Janelia and Google DeepMind and distributed under
Apache-2.0. We depend on the upstream project pinned to commit
`d015e9bfe441bd90ae431bac24c55cb74bdbce26`; its body assets remain upstream.
See [Whole-body physics simulation of fruit fly locomotion (Nature, 2025)](
https://doi.org/10.1038/s41586-025-09029-4).

Relevant upstream code: [flight environments](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/fly_envs.py),
[flight task](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/flight_imitation.py),
[synthetic trajectory](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/synthetic_trajectories.py).
