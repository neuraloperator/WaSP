# `config`

One config module per equation, plus a shared library of model configs. Configs are
[`zencfg`](https://github.com/neuraloperator/zencfg) classes: nested dataclass-like
declarations with defaults, resolved into a plain dict by
`make_config_from_cli(...).to_dict()` in the `train_{ac,ks,gl}.py` entry points. Any field
is overridable from the command line with `--section.key value`.

## Files

| File | Contents |
|------|----------|
| `ac_config.py` | `ACDefault`: Allen-Cahn equation, trainer, and FNO settings |
| `ks_config.py` | `KSDefault`: Kuramoto-Sivashinsky settings |
| `gl_config.py` | `GLDefault`: complex Ginzburg-Landau settings |
| `models.py` | Neural operator config classes shared with `neuralop` |

## Structure

Each `*Default` class holds the same sections, which `Trainer.setup` reads:

| Section | Consumer | Contents |
|---------|----------|----------|
| `model` | `Trainer.load_model` via `neuralop.get_model` | FNO channels, modes, padding, norm |
| `equation` | `Trainer.load_equation` | equation parameters, domain, resolution, loss weights, transport-map settings, dataset directory |
| `trainer` | `Trainer` | batch sizes, step budget, optimizer, LR schedule, replay buffer, normalizer, transport schedule |
| `wandb` | `WandBLogger` | entity, project, run name, group |
| `slurm` | `train_*.py` | submitit parameters, used when `--submit True` |
| `distributed` | `neuralop.training.setup` | DDP flags and seed |

Top-level flags: `verbose`, `debug` (disables W&B and checkpointing), `arch`, `submit`.

Constructor arguments are matched to config keys by name through
`wasp.training.utils.filter_kwargs`, so a key is picked up by the component whose signature
declares it, and unused keys are ignored.

## Per-equation differences

Everything not listed here is shared. The values below reproduce the paper.

| Key | AC | KS | CGLE |
|-----|----|----|------|
| `equation.domain` | `(5, 2π)` | `(0.128, 5)` | `(6.4, 50)` |
| `equation.resolution` | `(128, 128)` | `(128, 256)` | `(128, 256)` |
| `model.n_modes` | `[64, 128]` | `[64, 64]` | `[64, 64]` |
| `model.hidden_channels` | 32 | 32 | 64 |
| `model.data_channels` / `out_channels` | 3 / 1 | 3 / 1 | 5 / 2 |
| `trainer.dtype` | `float32` | `float32` | `complex64` |
| `trainer.n_steps` | 20000 | 100000 | 200000 |
| `trainer.max_grad_norm` | 5000 | 1000 | 10000 |
| `trainer.postprocess` | `{}` | `{"n_modes": 96}` | `{"n_modes": None}` |
| `trainer.buffer_size` | 5000 | 20000 | 20000 |
| `trainer.buffer_nsamples` | 32 | 16 | 16 |
| `trainer.transport_increment` | 0.005 | 0.002 | 0.002 |
| `trainer.transport_threshold` | 0.2 | 0.2 | 0.3 |
| `equation.ot_rank` | none | 64 | 96 |
| `equation.ot_eps` | none | 1e-2 | 1e-3 |

Allen-Cahn fits no transport map, so it has no `ot_*` or `fitting_samples` fields. The
higher spatial mode count comes from the sharp interfaces that phase separation produces at
large $t$.

## `models.py`

`ModelConfig` and its subclasses mirror the architectures `neuralop.get_model` accepts.
WaSP uses `SimpleFNOConfig` only, which each equation config subclasses as `FNO_2D` with
the channel counts, mode counts, domain padding, and normalization of the run. The other
classes (`FNOGNO*`, `GINO*`, `OTNO*`, and the preset FNO sizes) are carried along for
convenience and are not used by the experiments.
