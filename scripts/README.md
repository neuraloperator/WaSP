# `scripts`

Dataset generation, post-training evaluation, and batch launchers. Training itself is
launched from the `train_{ac,ks,gl}.py` entry points in the repository root.

## Files

| File | Purpose |
|------|---------|
| `datasets/create_{ac,ks,gl}_dataset.py` | CLI wrappers around `Equation.create_dataset` |
| `eval.py` | Post-training evaluation suite, writes `metrics.json` |
| `eval.sh` | Runs `eval.py` over every checkpoint of every method |
| `submit.sh` | Runs the training groups over all three equations |

## Dataset generation

Each script declares a `zencfg` config with the defaults used in the paper and calls the
equation's `create_dataset` class method, which solves the unforced equation with ETD-RK4
from random initial conditions and writes the trajectories to
`<root_dir>/<parameters>_<T>t@<Nt>_<L>x@<Nx>.pt`. The filename encodes the parameters, so
`Equation.setup` finds the file from the training config. Any field is overridable with
`--key value`.

```bash
python scripts/datasets/create_ks_dataset.py --nu 0.01 --n_samples 10000
```

For KS and CGLE the solver runs on a long time domain, discards `burn_in` steps, and cuts
the remainder into consecutive windows of the target resolution, so each window is a sample
of the invariant measure. Allen-Cahn produces one window per solve.

## Evaluation (`eval.py`)

```
main()
  ├── Trainer.load_checkpoint(checkpoint)        model, normalizer, equation, config
  ├── EqSolution.load(test_data_path)            held-out solver trajectories
  ├── test_set_eval()                            metric 1: solution relative L2
  ├── collect_coarse_rollouts()   ─┬─ attractor_orbit_eval()      metric 2: Sinkhorn W2
  │   (subsampled at delta_t)      ├─ energy_spectrum_eval()      metric 5
  │                                └─ spatial_correlation_eval()  metric 3
  ├── collect_fine_rollouts()      ─── temporal_correlation_eval() metric 4
  │   (native dt)
  ├── collect_long_rollouts()      ─── long_horizon_rollout_eval() extra diagnostic
  └── json.dump → <output_dir>/<wandb_name>/metrics.json
```

`autoregressive_rollout` drives both the model and the solver from the same initial
conditions: it predicts one window of length $T$, feeds the terminal slice back as the next
initial condition with zero forcing, and records snapshots after the burn-in. The paired
solver rollout gives the reference for metrics 3 to 5; a second, independent solver rollout
gives the calibration offset of the Sinkhorn estimator in metric 2.

Rollouts are expensive and are cached next to the metrics as `coarse_rollouts.pt`,
`fine_rollouts.pt`, and `long_rollouts.pt`. Delete them to recompute.

The Sinkhorn coupling behind metric 2 is O(n^2), so `attractor_orbit_eval` subsamples the
coarse rollout to at most `optimal_transport.num_samples` snapshots; the default (10,000)
equals the default coarse snapshot count, so no subsampling occurs in the paper's setting.
The entropic distance carries a finite-sample bias, so this value has to stay fixed for
numbers to be comparable across methods or against published results.

| Config section | Fields |
|----------------|--------|
| `coarse_rollout` | `snapshots`, `n_traj`, `burn_in`, `delta_t` |
| `fine_rollout` | `snapshots`, `n_traj`, `burn_in` |
| `long_rollout` | `snapshots`, `n_traj` |
| `optimal_transport` | `num_samples`, `eps`, `max_iters`, `stop_thresh` |

Each metric has a boolean flag (`test_set`, `attractor_dist`, `energy_spectrum`,
`spatial_correlation`, `temporal_correlation`, `long_horizon_rollout`). Metrics 2 to 5 need
an invariant measure, so they are turned off for Allen-Cahn. `long_horizon_rollout` is a
diagnostic and is not among the five metrics reported in the paper.

## Batch launchers

`submit.sh` takes a group index and runs that configuration over AC, KS, and CGLE:

| Index | Method | Run name |
|-------|--------|----------|
| 1 | WaSP | `wasp_{eq}` |
| 2 | MD, solution loss only | `md_{eq}` |
| 3 | MD+PINO, both losses frozen at $s = 0$ | `md_pino_{eq}` |
| 4 | PINO, PDE residual at $\nu_1$ | `pino_{eq}` |
| 5 | PINO+OT, PDE residual along the path | `pino_ot_{eq}` |

Group 5 also raises `transport_min_sample` from 1000 to 2000, so the schedule accumulates
twice as many samples before each decision.

`eval.sh` iterates over the same five run names per equation, skips checkpoints that are
absent, and passes the per-equation rollout and Sinkhorn settings. It reads test sets from
`./data/{ac,ks,gl}_testsets/`.
