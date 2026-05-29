# `wasp/training`

Training infrastructure for WaSP: the `Trainer` orchestrator, the `ReplayBuffer`, the
`TransportScheduler`, the `Normalizer`, and helpers for logging, checkpointing, and
optimization.

## Data Flow

```
Trainer.__call__(config)
       │
       ▼
   Trainer.setup(config)
       ├── load_model()          neuralop.get_model(config) → nn.Module
       ├── load_equation()       equations.{Name}(**config.equation)
       ├── load_optim()          optimizer_registry + scheduler_registry
       ├── load_normalizer()     Normalizer(solution_keys, condition_keys)
       ├── load_buffer()         ReplayBuffer(max_size, solution_shapes)
       │       └── equation.sample_solution() × init_buffer_size  (pre-fill)
       └── load_transport_scheduler()  TransportScheduler(...)
       │
       ▼
   Trainer.train()
       │
       └── per step: Trainer.train_step(s, bs_buffer, bs_gen)
               │
               ├── buffer.sample(bs_buffer)          → sol_buffer   (solution term)
               ├── equation.sample_solution(bs_gen)  → sol_gen
               ├── equation.forward(sol_gen)          → cond_gen
               ├── equation.map_condition(cond_gen, s) → cond_gen   (McCann interp)
               │
               ├── normalizer.normalize_cond(cond)
               ├── model(equation.prepare_input(norm_cond))
               ├── equation.postprocess(pred_u)
               ├── normalizer.denormalize_sol(sol_pred)
               │
               ├── equation.solution_loss(sol_buffer_pred, sol_buffer)  → buf_loss
               ├── equation.condition_loss(cond_gen_pred, cond_gen)     → gen_loss
               │
               ├── loss.backward() + optimizer.step() + scheduler.step()
               │
               ├── buffer.insert(sol_gen_pred, n=buffer_nsamples)
               ├── normalizer.update(sol_gen_pred, cond_gen)
               ├── transport_scheduler.step(gen_loss, bs_gen, step)
               └── logger.log(...) + logger.commit(step)
       │
       └── every eval_freq steps: Trainer.eval()
               ├── equation.map_to_test_condition(test_sol)
               ├── model forward pass on test conditions
               └── logger.log_images(...)
       │
       └── every save_freq steps: Trainer.save_checkpoint()
```

Both loss terms come from one forward pass on the concatenated batch. The solution term
regresses on pairs replayed from the buffer, the PDE residual term on freshly interpolated
inputs, and the predictions of the latter are inserted into the buffer for later steps.
Before the first transport step ($s = 0$) the exact manufactured solutions are inserted
instead of the predictions.

---

## `Trainer` (`trainer.py`)

Central orchestrator. Instantiated with no arguments; configured by calling
`Trainer()(config)` or restored from a checkpoint with `Trainer.load_checkpoint(path)`.

| Method | Description |
|--------|-------------|
| `__call__(config)` | Full training run: `setup` then `train` |
| `setup(config, from_checkpoint)` | Initialize all components from a config dict |
| `train()` | Main loop over `n_steps` steps |
| `train_step(s, bs_buffer, bs_gen)` | One step: sample, forward, loss, backward, buffer insert |
| `eval()` | Evaluation on the held-out test set (`no_grad`) |
| `visualize_buffer()` | Log images of the most recently inserted buffer samples |
| `save_checkpoint()` | Save model, normalizer, transport scheduler, and config |
| `load_checkpoint(path, override_config)` | Class method; restores trainer state from a `.pt` file |
| `load_model(config)` | Builds the FNO via `neuralop.get_model`, wraps in DDP if needed, casts dtype |
| `load_equation(config)` | Instantiates the equation class from the `equations` registry by name |
| `load_optim(config)` | Instantiates optimizer and scheduler from their registries |
| `load_normalizer(config)` | Builds `Normalizer` from the equation's normalizer keys |
| `load_buffer(config, fill_buffer)` | Builds `ReplayBuffer` and optionally pre-fills it |
| `load_transport_scheduler(config)` | Builds `TransportScheduler` from the trainer config |

A checkpoint holds `model` (state dict), `step`, `config`, `normalizer` (state dict), and
`scheduler` (transport scheduler state dict).

---

## `ReplayBuffer` (`replay_buffer.py`)

Circular FIFO buffer holding solution tensors in CPU memory. It stores the synthesized
solutions $\hat\mu_s = (G_\theta)_{\sharp} \nu_s$ and replays them as regression targets,
which removes the serial dependency between the two loss terms and keeps each target fixed
for several steps.

| Method | Description |
|--------|-------------|
| `insert(eq_solution, n)` | Write up to `n` samples; wraps around when full |
| `sample(batch_size, last)` | Draw `batch_size` random samples; `last=True` returns the most recent ones |
| `purge()` | Reset the buffer to empty |
| `pop(num_samples)` | Decrement the logical size without zeroing memory |
| `size` (property) | Number of valid samples |

Each field in `solution_shapes` is a pre-allocated tensor of shape
`(max_size, *field_shape)`. Insertion advances a pointer modulo `max_size`.

---

## `Normalizer` (`normalizer.py`)

Online EMA normalizer implemented as an `nn.Module` with persistent buffers. It tracks the
per-field mean square of the solution and input tensors and uses the root (RMS) to
normalize the inputs of $G_\theta$ and denormalize its outputs. The same statistics set the
relative scale of the two loss terms.

The update for field `key` with `n` new samples and momentum $m$:

$$\text{stat} \leftarrow (1-m)^n \cdot \text{stat} + \bigl(1 - (1-m)^n\bigr) \cdot \mathbb{E}[|x|^2]$$

The first update seeds the buffers directly. Later updates are gated by the `enabled` flag,
which the trainer sets from `TransportScheduler.adaptive`: outside the adaptive range of
$s$ the statistics are frozen, which keeps early out-of-distribution samples from
corrupting them and removes the noise floor of a drifting normalizer near $s = 1$.

| Method | Description |
|--------|-------------|
| `update(solution, condition, enabled)` | EMA update of the RMS statistics; returns a log dict |
| `normalize_cond(condition)` | Divide input fields by their RMS |
| `normalize_sol(solution)` | Divide solution fields by their RMS |
| `denormalize_sol(solution)` | Multiply solution fields by their RMS |
| `denormalize_cond(condition)` | Multiply input fields by their RMS |
| `stats` (property) | `{key: rms}` dict used to normalize the losses |

---

## `TransportScheduler` (`scheduler.py`)

Advances the McCann interpolation parameter $s \in [0,1]$ during training, combining a base
schedule that acts as a floor with an adaptive rule triggered by the normalized PDE residual loss.

**Two phases:**
1. **Warm-up** (first `end_pretraining` steps): $s$ is held at `start_s`.
2. **Transport**: once `min_sample` samples have accumulated, $s$ advances by `increment`
   if the mean normalized PDE residual loss is below `threshold` and $s$ lies strictly inside
   `adaptive_range`.
   Otherwise $s$ is raised to the base schedule (cosine or linear ramp from `start_s` to 1)
   whenever the schedule is ahead of it.

| Method / Property | Description |
|-------------------|-------------|
| `step(loss, numel, step)` | Accumulate the loss; possibly advance $s$ |
| `schedule_fn(step)` | Value of the base schedule at a given training step |
| `s` (property) | Current interpolation parameter |
| `adaptive` (property) | True when $s$ is strictly inside `adaptive_range` and the scheduler is enabled |
| `state_dict()` | Serialize `_s`, `loss_sum`, `n_samples` |
| `load_state_dict(state_dict)` | Restore from a checkpoint |

---

## `WandBLogger` (`loggers.py`)

Batching wrapper around `wandb.log`. Scalars and images accumulate internally and flush to
W&B every `log_every` steps, which cuts the API overhead.

| Method | Description |
|--------|-------------|
| `log(data, step)` | Append scalar values to the cache |
| `log_images(key, images, step, ...)` | Cache `wandb.Image` objects |
| `commit(step)` | Flush averaged scalars and images; reset the caches |
| `finish()` | Flush the remaining cache and call `wandb.finish()` |

Scalars are averaged over the `log_every`-step window. All `log()` and `log_images()` calls
inside a commit window must use the same `step`.

---

## `CheckpointManager` (`checkpointing.py`)

Wrapper around `torch.save` and `torch.load` that handles the `module.` prefix DDP adds to
state dicts.

| Method | Description |
|--------|-------------|
| `save(state_dict, step, name)` | Save to `{checkpoint_dir}/{name}/model_step_{step}.pt` |
| `load(path)` | Load a `.pt` file; uses `weights_only=False` for the full state |
| `load_model_state(model, state_dict)` | Load weights, stripping or adding `module.` as needed |
| `get_model_state(model)` | Extract the state dict, unwrapping DDP if present |

---

## `Registry` and Helpers (`utils.py`)

### `Registry`

Name-to-class registry with a fallback Python module (for example `torch.optim`).

| Method | Description |
|--------|-------------|
| `register(name)` | Decorator registering a class under a string key |
| `get(name)` | Retrieve by name; falls back to `fallback_module` |

Two registries are preconfigured:
- `optimizer_registry`: all `torch.optim` classes
- `scheduler_registry`: `"CosineWithWarmupLR"` plus all `torch.optim.lr_scheduler` classes

### `filter_kwargs(obj, config, exclude)`

Returns the keys of `config` that match parameter names in the signature of `obj`. Used by
the `Trainer.load_*` methods to pass config dicts to constructors without selecting keys by
hand.

### `update_config(config, override)`

Recursively merges `override` into `config`; nested dicts are merged rather than replaced.
Used by `Trainer.load_checkpoint` to override a restored config.

### `CosineWithWarmupLR`

Registered factory returning a `SequentialLR` that applies linear warmup (`LinearLR`)
followed by cosine annealing (`CosineAnnealingLR`). Parameters: `n_steps`, `eta_min`,
`warmup_steps`, `warmup_factor`. The shipped configs use `StepLR` instead.
