# Autonomous connectome training

Run from the repository root after installing `.[superfly,dev]` and setting `MUJOCO_GL=egl`:

```bash
python -m fly_fruit_fly.flight_training --rounds 12 --output runs/student
fly assess runs/student/evaluation/metrics.json --require-pass
```

The [Actions workflow](../.github/workflows/flight-student.yml) executes the same training path.

1. Materialize 256 BANC v888 neurons with measured, fixed recurrent connections.
2. Collect four released-expert demonstrations and train sensory/dynamics/motor parameters for 12 imitation epochs.
3. Freeze observation normalization and recurrent features. Fit the motor readout by regularized least squares against the expert's pre-tanh actions.
4. Compare the existing readout and two ridge penalties on three autonomous validation episodes.
5. Collect two corrective demonstrations on mixed teacher/student states, gradually removing teacher assistance, and refit using the accumulated dataset.
6. Keep the best validation checkpoint. Stop when all three validation episodes pass, or after the bounded round budget.
7. Evaluate that checkpoint once on ten separate held-out seeds, with no expert actions, and record the real simulator movie.

Validation seeds are 30000–30002. Final test seeds are 70000–70009. Training/corrective seeds are separate. Initial wingbeat phase is discrete, so different seeds can share phases; this does not establish broad out-of-distribution generalization.

`training.json` records all candidate scores, corrective rollouts, and checkpoint selection. `student.pt` and `graph.npz` are the selected model and matching graph. `evaluation/` holds the held-out metrics, trajectory and movie; `assessment.json` applies the unchanged flight gate.

The ridge fit trains only the motor readout at that stage. It reads recurrent connectome activity, not raw observations or live expert actions. Biological connectivity remains an engineering structural prior; this procedure does not establish an advantage over a shuffled graph.

