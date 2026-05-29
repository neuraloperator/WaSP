# `wasp/equations`

PDE abstractions used by WaSP: a shared data model (`EqData`, `EqSolution`, `EqCondition`),
an abstract interface (`EquationBase`), and three equations (`ACEquation`, `KSEquation`,
`GLEquation`).

Every equation is written as $N(u) = v$ with $v = (u(0,x), f(t,x))$, so `forward` applies
the PDE operator to a solution and `backward` solves the equation with an ETD-RK4 scheme.

## Data Flow

```
create_dataset()         (offline, writes .pt to disk)
       |
       v
EqSolution.load()        (loads tensor_dict + domain from .pt)
       |
       v
equation.setup()         (fits GRF power spectrum + transport map T from target data)
       |
   ┌───┴───────────────────────────────────────────────────────────────────┐
   │ per training step                                                     │
   │                                                                       │
   │  sample_solution(n)   →   EqSolution  (u sampled from μ₀)             │
   │         │                                                             │
   │         ▼                                                             │
   │     forward(sol)      →   EqCondition (v = N(u): f + ic)              │
   │         │                                                             │
   │         ▼                                                             │
   │  map_condition(cond, s) → EqCondition (McCann interp toward ν₁)       │
   │         │                                                             │
   │         ▼                                                             │
   │  prepare_input(cond)  →   Tensor      (model input: [f, ic, pe])      │
   │         │                                                             │
   │         │  (model forward pass happens here)                          │
   │         ▼                                                             │
   │  postprocess(u_pred)  →   dict[str, Tensor]  (low-pass filter)        │
   │         │                                                             │
   │         ▼                                                             │
   │  solution_loss() / condition_loss()  →  (loss, log_dict)              │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
       |
       v
   backward(cond)         (ETD-RK4 solver, used for datasets and eval rollouts)
       |
       v
   visualize(sol, cond)   (returns numpy arrays for W&B image logging)
```

## Data Classes (`equation_data.py`)

### `EqData`

Container for batched PDE fields. Holds a `tensor_dict: dict[str, Tensor]` whose tensors
all have shape `(n_samples, *field_shape)`, and a `domain: tuple[float, ...]` giving the
physical extent.

| Method | Description |
|--------|-------------|
| `__getitem__(str)` | Returns the named tensor |
| `__getitem__(int/slice/Tensor)` | Returns a sub-batched `EqData` |
| `to(device, dtype)` | Moves all tensors, returns a new instance |
| `from_list(data_list)` | Concatenates a list of `EqData` along the batch axis |
| `new_data(tensor_dict)` | New instance with updated tensors, same domain |
| `save(data_list, path)` | Concatenates and saves to a `.pt` file (CPU tensors) |
| `load(path)` | Loads a file written by `save()` |
| `keys` (property) | Field names in insertion order |

### `EqSolution(EqData)`

Solution fields. For all three equations the key is `"u"` with shape `(n, nt, nx)`, complex
for CGLE. Adds `new_solution(tensor_dict)` as a typed alias for `new_data`.

### `EqCondition(EqData)`

Input fields $v$, named the condition in the code. The keys are `"f"` (forcing, shape `(n, nt, nx)`) and `"ic"` (initial
condition, shape `(n, nx)`). Adds `new_condition(tensor_dict)` as a typed alias.

---

## Abstract Interface (`equation_base.py`)

### `EquationBase`

All equations inherit from this class and implement every `@abstractmethod`.

**Class attributes** (set by the subclass `__init__`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `domain` | `tuple[float, ...]` | Physical extents `(T, L)` |
| `resolution` | `tuple[int, ...]` | Grid sizes `(nt, nx)` |
| `solution_shapes` | `dict[str, tuple]` | Per-key shape of solution tensors, without the batch dim |
| `condition_shapes` | `dict[str, tuple]` | Per-key shape of input (condition) tensors, without the batch dim |
| `solution_normalizer_key` | `list[str]` | Solution keys whose RMS the normalizer tracks |
| `condition_normalizer_key` | `list[str]` | Input keys whose RMS the normalizer tracks |
| `test_data` | `dict[str, EqData]` | Held-out test set, populated by `setup()` |

**Abstract methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `setup` | `(device) -> None` | Load the test set, fit the power spectrum of $\mu_0$, fit the transport map |
| `forward` | `(EqSolution) -> EqCondition` | Apply the PDE operator, $v = N(u)$ |
| `backward` | `(EqCondition, ...) -> EqSolution` | Solve $u = G(v)$ with ETD-RK4 |
| `sample_solution` | `(n, device) -> EqSolution` | Draw $u \sim \mu_0$ |
| `map_condition` | `(EqCondition, s) -> EqCondition` | McCann interpolation at $s \in [0,1]$ |
| `map_to_test_condition` | `(EqSolution) -> EqCondition` | Turn a solution into a zero-forcing target input |
| `prepare_input` | `(EqCondition) -> Tensor` | Stack the fields into the model input |
| `postprocess` | `(Tensor, ...) -> dict[str, Tensor]` | Post-process the raw model output |
| `solution_loss` | `(pred, gt, normalizers, stage) -> (Tensor, dict)` | $\alpha_u$ terms (manufactured-data loss) |
| `condition_loss` | `(pred, gt, normalizers, stage) -> (Tensor, dict)` | $\alpha_v$ terms (PDE residual loss) |
| `create_dataset` | `(cls, ...) -> None` | Generate and save a dataset with the solver |
| `visualize` | `(sol, cond, normalizers) -> (ndarray, ndarray)` | Colormap images for W&B |

`EquationBase` also provides `_solve_coeff(device, is_complex)`, which computes and caches
the six ETD-RK4 coefficient tensors `(E, E2, q, f1, f2, f3)`. Subclasses must set
`self._etdrk4_L`, `self._etdrk4_dt`, and `self._etdrk4_M` before calling it.

---

## Concrete Equations

### `ACEquation` (`ac_equation.py`)

Allen-Cahn: $u_t = \nu u_{xx} + u - u^3 + f(t,x)$.

**Constructor parameters** (set under `equation` in `config/ac_config.py`):

| Parameter | Description |
|-----------|-------------|
| `nu` | Diffusion coefficient |
| `ic_amplitude` | Amplitude of the initial-condition power spectrum |
| `power_decay` | Spectral decay rate $\gamma$: $P(k) \propto e^{-\gamma k}$ |
| `domain` | `(T, L)` |
| `resolution` | `(nt, nx)` |
| `loss_weights` | Per-term weights (`sol`, `sol_ic`, `cond`, `cond_ic`, `cond_h1`) |
| `test_samples` | Number of test samples held out from the dataset file |

**Notes:**
- `setup`: builds the initial-condition power spectrum analytically. No transport map is
  fitted, since the manufactured initial conditions are already drawn from the target
  distribution.
- `sample_solution`: samples the initial-condition GRF, then takes one Lie-Trotter
  splitting step of size $t$ for each output time: the reaction ODE $u_t = u - u^3$ is
  solved in closed form pointwise, and the diffusion semigroup is applied exactly in
  Fourier space.
- `forward`: computes $f = u_t - \nu u_{xx} + u^3 - u$, with `FiniteDifference` for $u_t$
  and spectral differentiation for $u_{xx}$.
- `map_condition`: identity on the initial condition, $f \mapsto (1-s) f$ on the forcing.
- `postprocess`: no filtering; the single output channel is unpacked.
- `backward`: ETD-RK4, solved spectrally.

### `KSEquation` (`ks_equation.py`)

Kuramoto-Sivashinsky: $u_t = -u u_x - u_{xx} - \nu u_{xxxx} + f(t,x)$.

**Constructor parameters** (set under `equation` in `config/ks_config.py`):

| Parameter | Description |
|-----------|-------------|
| `nu` | Hyperviscosity coefficient |
| `domain` | `(T, L)` |
| `resolution` | `(nt, nx)` |
| `loss_weights` | Per-term weights |
| `ot_rank` | PCA dimension used by the Sinkhorn transport map |
| `fitting_samples` | Samples used to fit the transport map and the power spectrum |
| `test_samples` | Held-out test samples |
| `ot_eps` | Sinkhorn entropy regularization |
| `ot_max_iters` | Maximum Sinkhorn iterations |
| `ot_stop_thresh` | Sinkhorn convergence threshold |

**Notes:**
- `setup`: fits a separable space-time power spectrum, taking the spatial factor from the
  target initial conditions and calibrating a Gaussian temporal factor to the time
  derivative the unforced equation gives at $t = 0$. Then fits the transport map $T$ from
  the manufactured initial conditions to the target ones with `OTMapping`.
- `sample_solution`: draws from the fitted spectrum with `sample_grf`, on a doubled
  temporal grid whose first half is kept.
- `forward`: computes $f = u_t + u u_x + u_{xx} + \nu u_{xxxx}$, with `FiniteDifference` for
  $u_t$ and spectral differentiation for the spatial terms. Casts to `float64` for FFT
  accuracy.
- `map_condition`: $\mathrm{ic} \mapsto (1-s)\mathrm{ic} + sT(\mathrm{ic})$,
  $f \mapsto (1-s) f$.
- `postprocess`: low-pass filter retaining `n_modes` spatial Fourier modes. The $u_{xxxx}$
  term of the residual scales as $k^4$, which is numerically unstable in the top modes.
- `backward`: ETD-RK4 with de-aliasing above $N_x/3$.

### `GLEquation` (`gl_equation.py`)

Complex Ginzburg-Landau: $u_t = u + (1+i\alpha)u_{xx} - (1+i\beta)|u|^2 u + f(t,x)$.

**Constructor parameters** (set under `equation` in `config/gl_config.py`):

| Parameter | Description |
|-----------|-------------|
| `alpha`, `beta` | CGLE coefficients (defect turbulence at $\alpha = 2$, $\beta = -2$) |
| `domain`, `resolution` | `(T, L)` and `(nt, nx)` |
| `loss_weights` | Per-term weights |
| `ot_rank`, `fitting_samples`, `test_samples` | As in `KSEquation` |
| `ot_eps`, `ot_max_iters`, `ot_stop_thresh` | Sinkhorn parameters |

**Notes:**
- Fields are complex (`complex64`), so the transport map and the model see the real and
  imaginary parts as two real channels.
- `setup`: as in `KSEquation`, and the spatial spectrum doubles as the distribution the
  plane-wave wavenumber is drawn from.
- `sample_solution`: draws an unstable plane wave whose wavenumber follows that spectrum
  and whose frequency follows the CGLE dispersion relation
  $\omega(k) = -\beta - (\alpha - \beta)k^2$, then modulates it by a complex GRF. A plain
  GRF would omit the plane-wave structure the defect-turbulence dynamics are built from.
- `forward`: computes $f = u_t - u - (1+i\alpha)u_{xx} + (1+i\beta)|u|^2 u$ spectrally.
- `backward`: complex ETD-RK4; `_solve_coeff(is_complex=True)` adjusts the phase of the
  contour integral.
- `visualize`: colors the field by phase and magnitude.
