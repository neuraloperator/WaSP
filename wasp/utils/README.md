# `wasp/utils`

Three self-contained utilities used by the equation classes: an entropy-regularized
optimal transport map (`OTMapping`), a high-order finite difference operator
(`FiniteDifference`), and functional helpers for GRF sampling and norms.

## Data Flow

```
equation.setup()
    │
    ├── OTMapping.fit(manufactured_ics, target_ics)
    │       └── Sinkhorn iterations → dual potentials (u_i, v_j) stored on self
    │
    └── [GRF power spectrum fitted from the target data]
    │
equation.sample_solution()
    │
    └── sample_grf(power, batch_size, resolution) → Tensor  (u ~ μ₀)
    │
equation.forward(solution)
    │
    ├── FiniteDifference.derive(u, delta, order=1, dim=1)  → ∂_t u
    └── spectral spatial derivatives (inline, per equation)
    │
equation.map_condition(cond, s)
    │
    └── OTMapping.map(ic) → T(ic)   (barycentric projection onto the target support)
    │
equation.solution_loss() / condition_loss()
    │
    ├── l2_norm(error)   → scalar (batch-summed, resolution-independent)
    └── h1_norm(error)   → scalar (L2 + spectral ∂_x, batch-summed)
```

---

## `OTMapping` (`optimal_transport.py`)

Entropy-regularized transport map fitted with the Sinkhorn algorithm and applied by
barycentric projection, following
[Pooladian and Niles-Weed, 2021](https://arxiv.org/abs/2109.12004). The iteration runs in
the log domain on PyKeOps lazy tensors, so the $n \times m$ coupling is never held in
memory.

The cost is $C(x,y) = \tfrac{1}{2}\lVert x - y\rVert^2$, so `fit` returns a $W_2$ distance.
After fitting, `map(x)` projects new source points onto the target support:

$$T(x_i) = \frac{\sum_j P_{ij} y_j}{\sum_j P_{ij}}, \qquad \log P_{ij} = \frac{-C_{ij} + u_i + v_j}{\varepsilon}.$$

**PCA acceleration** (`q`): both clouds are projected onto the top `q` principal components
of `y` before the Sinkhorn iteration, which reduces the cost dimension from $d$ to $q$.
This matters when $d$ is large, for instance $N_x = 256$ initial-condition snapshots. A
warning is raised when the projection explains less than 99% of the variance of `y`. Both
clouds are also rescaled by the RMS radius of `y`, so `eps` is dimensionless.

### Constructor

```python
OTMapping(
    eps=1e-3,           # Sinkhorn entropy regularization
    max_iters=100,      # maximum Sinkhorn iterations
    stop_thresh=1e-5,   # convergence threshold on the max change in the potentials
    q=None,             # PCA dimension; None disables PCA
    verbose=False,
    device=torch.device("cpu"),
    dtype=torch.float32,
)
```

### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `(x: [n,d], y: [m,d], return_se=False) -> float` | Run Sinkhorn from source `x` to target `y`, store the dual potentials, and return the $W_2$ distance. With `return_se`, also returns its standard error from the variance of the potentials. |
| `map` | `(x: [n,d]) -> [n,d]` | Barycentric projection of new source points. Requires `fit` first. |

**Use in WaSP**: `fit` is called once per equation in `setup()` to estimate
$T: \nu_0 \to \nu_1$ over initial-condition snapshots; `map` is called every training step
inside `map_condition()`. `scripts/eval.py` reuses `fit` for the invariant-measure
distance.

---

## `FiniteDifference` (`differentiation.py`)

High-order finite difference stencil for time differentiation. The coefficients are solved
at construction time as a polynomial least-squares fit (Vandermonde system) with SymPy, for
every derivative order up to `degree`.

Three stencil families are precomputed: central for interior points, forward for the
`N//2` points at the left edge, and backward for the `N//2` points at the right edge. The
one-sided stencils keep $\partial_t u$ accurate at the temporal boundaries, which have no
periodicity to exploit.

### Constructor

```python
FiniteDifference(
    N=5,       # stencil width (must be odd)
    degree=4,  # polynomial degree to fit (default: N-1)
)
```

### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `derive` | `(u, delta, order=1, dim=0, boundary="none") -> Tensor` | Apply the stencil along `dim` with grid spacing `delta`. Boundary modes: `"none"` (one-sided stencils at both edges), `"periodic"` (circular padding), `"even"` / `"odd"` (reflection at the left edge). |

**Use in WaSP**: all three equations call
`FiniteDifference(N=5, degree=4).derive(u, delta=T/nt, dim=1)` for $\partial_t u$ when
evaluating the PDE operator in `forward()`.

---

## Functional Utilities (`utils.py`)

### `sample_grf`

```python
sample_grf(
    power: Tensor,        # power spectrum, one dimension per resolution axis
    batch_size: int,
    resolution: tuple[int, ...],
    complex_valued: bool = False,
    device: torch.device = cpu,
) -> Tensor               # shape (batch_size, *resolution)
```

Draws Gaussian random fields with the given power spectrum. Sampling happens in Fourier
space: complex Gaussian noise is weighted by $\sqrt{P(k)/2}$ per mode and transformed back
with `irfftn` (real) or `ifftn` (complex). Every equation uses it in `sample_solution()`
to build $\mu_0$ or part of it: the full field for KS, the initial condition for AC, and
the perturbation for CGLE.

### `l2_norm`

```python
l2_norm(x: Tensor) -> Tensor  # scalar, summed over the batch
```

Computes $\sum_{i=1}^B \lVert x_i\rVert_2 / \sqrt{D}$, where $D$ is the number of elements
per sample. Dividing by $\sqrt{D}$ makes the value independent of the grid, so loss
magnitudes are comparable across resolutions.

### `h1_norm`

```python
h1_norm(x: Tensor, l: float = 1.0) -> Tensor  # scalar, summed over the batch
```

Computes

$$\|x\|_{L^2} + \|\partial_x x\|_{L^2}$$

with the spatial derivative taken spectrally along the last dimension, for real and complex
tensors alike. Used by the `cond_h1` term of `condition_loss()`, which is off by default
(`loss_weights.cond_h1 = 0`).
