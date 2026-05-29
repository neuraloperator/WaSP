## WaSP: Wasserstein Data Synthesis Pipeline

This repository implements *WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free 
Operator Learning*.

WaSP trains neural operators on PDEs without a solver. Supervised operator learning needs
input-solution pairs from a numerical solver, which is the cost the surrogate is meant
to remove. WaSP instead manufactures exact pairs by applying the PDE operator to sampled
solutions, and moves the input distribution of those pairs along a Wasserstein
interpolation toward the test-time distribution. Training becomes a sequence of supervised
problems, breaking the one large distribution shift into manageable pieces.

- [Problem Setting](#problem-setting)
- [Why Existing Approaches Fall Short](#why-existing-approaches-fall-short)
- [The WaSP Algorithm](#the-wasp-algorithm)
- [Supported PDEs](#supported-pdes)
- [Methods and Baselines](#methods-and-baselines)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

> [!NOTE]
> **Input and condition are the same object.** The paper calls the PDE input
> $v = (u(0,x), f(t,x))$, initial state and forcing, the *input*. The code calls it the
> *condition*: `EqCondition`, `map_condition`, `condition_loss`, the `cond*` loss weights,
> and the `cond_loss` logs all refer to the input $v$, and the condition loss is the PDE
> residual loss $\ell_{\mathrm{PDE}}$.

### Problem Setting

A PDE is written as a bijection $N: \mathcal{U} \to \mathcal{V}$ between a space of
solutions and a space of inputs,

$$N(u) = v, \qquad G(v) = u,$$

where $u \in \mathcal{U}$ is the solution, $v \in \mathcal{V}$ is the input (initial
state and forcing), and $G = N^{-1}$ is the solution operator. The goal is to learn a
neural operator $G_\theta$ that minimizes

$$\mathbb{E}_{v \sim \nu_1} \| G_\theta(v) - G(v) \|_{\mathcal{U}},$$

where $\nu_1 \in \mathcal{P}(\mathcal{V})$ is the target input distribution. Training
is *solver-free*; that is, no solver-generated pair $(v, G(v))$ with $v \sim \nu_1$ is 
used as a label. Unpaired inputs $v \sim \nu_1$ are available, as they are to any
physics-informed method.

### Why Existing Approaches Fall Short

**PINO** constrains $G_\theta$ to be a right inverse of $N$ by minimizing the residual

$$\mathbb{E}_{v \sim \nu} \| (N \circ G_\theta)(v) - v \|_{\mathcal{V}}.$$

Two difficulties follow. A residual measured in a weak norm need not control the solution
error, and a norm strong enough to control it inflates the variance of the mini-batch
gradient. $N$ also enters the computation graph, so nonlinear terms and high-order
derivatives make the objective hard to optimize. This is what we called "regularity--variance
tradeoff" in our paper.

**Manufactured data (MD)** samples solutions $u \sim \mu_0$ from a prior, applies the PDE
operator to get $v = N(u)$, and minimizes the supervised loss

$$\mathbb{E}_{u \sim \mu_0} \| (G_\theta \circ N)(u) - u \|_{\mathcal{U}}.$$

Every pair is exact and the problem is a well-conditioned regression, but the induced
input distribution $\nu_0 = N_{\sharp} \mu_0$ can be mutually singular with $\nu_1$.

### The WaSP Algorithm

WaSP splits the gap between $\nu_0$ and $\nu_1$ into small steps along the McCann
interpolation, and keeps the supervision exact at every step.

**Step 1.** Choose a manufactured solution distribution $\mu_0 \in \mathcal{P}(\mathcal{U})$
that is cheap to sample, and set $\nu_0 = N_{\sharp} \mu_0$.

**Step 2.** Estimate a transport map $T$ from $\nu_0$ to $\nu_1$ with entropy-regularized
Sinkhorn and barycentric projection. This is a one-time preprocessing step.

**Step 3.** Warm up on the exact manufactured pairs at $s = 0$.

**Step 4.** For $s \in (s_1, s_2, s_3, \dots, 1)$:
- Interpolate the input distribution:
  $\nu_s = \bigl((1-s)\mathrm{Id} + sT\bigr)_{\sharp} \nu_0$
- Synthesize solutions with the current operator:
  $\hat\mu_s = (G_{\theta})_{\sharp} \nu_s$
- Apply $N$ to recover the exact input of each synthesized solution, and take a
  gradient step on the pairs.

Each step minimizes

$$L(\theta,\mu,\nu) = \alpha_u \mathbb{E}_{u \sim \mu}\|( G_\theta \circ N)(u) - u\| + \alpha_v \mathbb{E}_{v \sim \nu}\|(N \circ G_\theta)(v) - v\| .$$

The implementation measures both terms with the unsquared, grid-normalized $L^2$ norm
(`wasp.utils.l2_norm`) rather than its square. The first term is the solution loss
$\ell_{\mathrm{MD}}$ on manufactured or synthesized pairs and carries the primary learning
signal. The second term is the PDE residual $\ell_{\mathrm{PDE}}$, and it bounds the distribution shift
that the synthesis step introduces: the gap $W(\nu_s, (N \circ G_\theta)_{\sharp} \nu_s)$
between the intended and realized input distributions is at most the residual over
$\nu_s$. The residual is used as a local regularizer and as the trigger for advancing $s$,
never as a certificate of the solution error, so a weak norm suffices. It also blocks the
confirmation bias of self-generated targets, because the residual term regresses on a
target the model did not produce.

**Structure of the input space.** Each input $v = (u_0, f)$ pairs an initial state
$u_0 = u(0, \cdot)$ with a forcing field $f$. The target forcing distribution is the point
mass at $f = 0$, so any transport to $\nu_1$ sends the forcing to zero and the map
factors: $T$ acts on the initial condition only, and the forcing anneals linearly. For
$v^{(0)} = (u_0^{(0)}, f^{(0)}) \sim \nu_0$,

$$v_s = \bigl(((1-s)\mathrm{Id} + s T)(u_0^{(0)}),\quad (1-s) f^{(0)}\bigr).$$

Optimal transport is therefore solved on the spatial grid, not the space-time grid.

**Curriculum.** At small $s$ the forcing is large and the solution is largely determined by
it, so the operator inverts a well-conditioned problem. As $s \to 1$ the forcing recedes
and the initial state moves onto the target attractor, so the intrinsic dynamics of the
equation have to be learned. The interpolation presents a sequence of progressively harder
problems.

---

### Supported PDEs

| Acronym | Equation | Config | Script |
|---------|----------|--------|--------|
| **AC** | Allen-Cahn: $u_t = \nu u_{xx} + u - u^3 + f(t,x)$ | `config/ac_config.py` | `train_ac.py` |
| **KS** | Kuramoto-Sivashinsky: $u_t = -u u_x - u_{xx} - \nu u_{xxxx} + f(t,x)$ | `config/ks_config.py` | `train_ks.py` |
| **CGLE** | Complex Ginzburg-Landau: $u_t = u + (1+i\alpha)u_{xx} - (1+i\beta)\vert u\vert^2 u + f(t,x)$ | `config/gl_config.py` | `train_gl.py` |

All three are one-dimensional evolution equations with periodic spatial boundaries. The
forcing $f$ is added so that every sufficiently regular field $u$ is the unique solution of
$N(u) = v$ with $v = (u(0,x), f(t,x))$, which realizes the bijection above. The target
distribution $\nu_1$ is the unforced equation, $f \equiv 0$.

**Allen-Cahn** ($\nu = 0.01$, $\Omega = [0,5]_t \times [0,2\pi]_x$): a reaction-diffusion
equation that separates into the phases $u = \pm 1$. Initial conditions are drawn from a
Gaussian random field with $P(k) \propto e^{-k}$. This is the mildest benchmark.

**Kuramoto-Sivashinsky** ($\nu = 0.01$, $\Omega = [0,0.128]_t \times [0,5]_x$):
spatio-temporal chaos. The lowest seven Fourier modes on this domain are linearly
unstable and the sixth grows fastest. Rescaling maps the equation to the standard KS
equation at domain length $L/\sqrt{\nu} = 50$, well above the chaotic threshold. Initial
conditions are drawn from the invariant measure.

**Complex Ginzburg-Landau** ($\alpha = 2$, $\beta = -2$,
$\Omega = [0,6.4]_t \times [0,50]_x$): a complex field in the *defect turbulence* regime,
with persistent disorder and phase singularities ($|u| = 0$). Initial conditions are drawn
from the invariant measure.


### Methods and Baselines

The five configurations reported in the paper share the optimizer, schedule, step budget,
and total batch size of 64, and differ only in the transport schedule and in how the batch
is split between the two loss terms.

| Method | `transport_enabled` | `batch_size_buffer` | `batch_size_gen` | Objective |
|--------|--------------------|--------------------|------------------|-----------|
| **WaSP** (ours) | True | 32 | 32 | Both terms along the path $\nu_s$ |
| **PINO** | False | 0 | 64 | PDE residual at $\nu_1$, with $s$ fixed at 1 |
| **PINO+OT** | True | 0 | 64 | PDE residual along the path $\nu_s$ |
| **MD** | False | 64 | 0 | Solution loss on the fixed $\mu_0$ |
| **MD+PINO** | False | 32 | 32 | Both terms frozen at $(\mu_0, \nu_0)$ |

PINO and PINO+OT raise `cond_ic` from 0.1 to 1.0, so that the interior and
initial-condition residuals are weighted equally; without the change PINO ignores the
initial condition and training degenerates. The two ablations isolate the two components of
WaSP: PINO+OT keeps the transport curriculum and drops the self-generated supervision,
MD+PINO keeps the combined objective and drops the curriculum.

---

### Installation

Clone the repository:
```bash
git clone git@github.com:neuraloperator/WaSP.git
```

Create the environment:
```bash
conda create -n wasp python=3.13
conda activate wasp
```

Install torch and WaSP:
```bash
# Adjust --index-url to match your CUDA version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

`pykeops` is sensitive to the CUDA and torch versions. If installation fails, check
`CUDA_HOME`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` before reinstalling. For development,
install the hook:
```bash
pre-commit install
```

---

### Data Preparation

Datasets are not shipped with the repository and have to be generated once. Each equation
has a `create_dataset()` class method that runs the ETD-RK4 solver from random initial
conditions with no forcing and writes the trajectories to disk. The filename encodes the
equation parameters and the resolution, so `setup()` locates it from the training config.
`scripts/datasets/create_{ac,ks,gl}_dataset.py` expose the same parameters as a CLI through
`zencfg`; any field can be overridden with `--key value`.

The dataset serves two purposes. `test_samples` trajectories are held out for evaluation.
For KS and CGLE, `fitting_samples` further trajectories supply the unpaired target
inputs used to fit the power spectrum of $\mu_0$ and the transport map; Allen-Cahn
needs neither, since its initial-condition spectrum is analytic and its transport map is
the identity. No solution from this file is ever used as a training label.

For the two chaotic equations the solver runs on a long time domain, the first `burn_in`
steps are discarded, and the remainder is cut into consecutive windows of the target
resolution, so every window is a sample of the invariant measure.

#### Kuramoto-Sivashinsky

```bash
python scripts/datasets/create_ks_dataset.py --nu 0.01 --n_samples 10000 --root_dir ./data/ks_datasets
```

Or through the Python API:

```python
from wasp.equations.ks_equation import KSEquation

# Generates ./data/ks_datasets/nu_1e-02_0.128t@128_5.0x@256.pt
KSEquation.create_dataset(
    nu=0.01,
    n_samples=10000,
    batch_size=10,
    domain=(0.128, 5.0),            # target (T, L)
    resolution=(128, 256),          # target (N_t, N_x)
    solver_domain=(10.0, 5.0),      # solver time domain, long enough for burn-in
    solver_resolution=(20000, 512),
    burn_in=1000,                   # solver steps discarded
    root_dir="./data/ks_datasets",
    verbose=True,
)
```

#### Allen-Cahn

```bash
python scripts/datasets/create_ac_dataset.py --nu 0.01 --n_samples 1000 --root_dir ./data/ac_datasets
```

```python
from wasp.equations.ac_equation import ACEquation

# Generates ./data/ac_datasets/nu_1e-02_amp_1e-03_decay_1.0e+00_5.0t@128_6.283185307179586x@128.pt
ACEquation.create_dataset(
    nu=0.01,
    ic_amplitude=1e-3,   # amplitude of the initial-condition power spectrum
    power_decay=1.0,     # P(k) proportional to exp(-k)
    n_samples=1000,
    batch_size=10,
    domain=(5.0, 6.283185307179586),  # (T, L) = (5, 2 pi)
    resolution=(128, 128),
    solver_resolution=(256, 256),
    root_dir="./data/ac_datasets",
)
```

Allen-Cahn is not chaotic, so it needs no burn-in: each trajectory is one sample and the
solver runs directly on the target time domain.

#### Complex Ginzburg-Landau

```bash
python scripts/datasets/create_gl_dataset.py --alpha 2.0 --beta -2.0 --n_samples 10000 --root_dir ./data/gl_datasets
```

```python
from wasp.equations.gl_equation import GLEquation

# Generates ./data/gl_datasets/alpha_2e+00_beta_-2e+00_6.4t@128_50.0x@256.pt
GLEquation.create_dataset(
    alpha=2.0,
    beta=-2.0,
    n_samples=10000,
    batch_size=10,
    domain=(6.4, 50.0),
    resolution=(128, 256),
    solver_domain=(500.0, 50.0),
    solver_resolution=(20000, 512),
    burn_in=1000,
    root_dir="./data/gl_datasets",
    verbose=True,
)
```

---

### Training

**WaSP on KS** (100k steps):
```bash
python train_ks.py --wandb.name wasp_ks
```

Checkpoints are written every `save_freq` steps to `./checkpoints/wasp_ks/`. Evaluation on
the held-out test set runs every `eval_freq` steps during training.

**PINO** (PDE residual loss at $\nu_1$):
```bash
python train_ks.py \
    --wandb.name pino_ks \
    --trainer.batch_size_buffer 0 \
    --trainer.batch_size_gen 64 \
    --trainer.transport_start_s 1.0 \
    --trainer.transport_enabled False \
    --equation.loss_weights.cond 1.0 \
    --equation.loss_weights.cond_ic 1.0
```

**PINO+OT** (PDE residual loss along the path):
```bash
python train_ks.py \
    --wandb.name pino_ot_ks \
    --trainer.batch_size_buffer 0 \
    --trainer.batch_size_gen 64 \
    --equation.loss_weights.cond 1.0 \
    --equation.loss_weights.cond_ic 1.0 \
    --trainer.transport_min_sample 2000
```

**MD** (manufactured solution loss on the fixed $\mu_0$):
```bash
python train_ks.py \
    --wandb.name md_ks \
    --trainer.transport_enabled False \
    --trainer.batch_size_buffer 64 \
    --trainer.batch_size_gen 0
```

**MD+PINO** (both terms frozen at $s = 0$):
```bash
python train_ks.py --wandb.name md_pino_ks --trainer.transport_enabled False
```

Replace `ks` and `train_ks.py` with `ac` and `train_ac.py` or `gl` and `train_gl.py` for
the other equations. Full hyperparameters for every run are listed in the appendix of 
the paper.

---

### Evaluation

`scripts/eval.py` measures pointwise accuracy and the fidelity of the long-time statistics
through five metrics:

1. **Solution relative $\ell_2$**, on the test set with zero forcing:
$$\frac{\mathbb{E}_{v \sim \nu_1} \|G_\theta(v) - G(v)\|_2}{\mathbb{E}_{v \sim \nu_1} \|G(v)\|_2}$$
2. **Invariant-measure distance**: Sinkhorn $W_2$ between the snapshot ensembles of the
   model and the solver, both collected from autoregressive rollouts on the attractor. A
   second solver rollout gives the reference offset of the estimator. The coupling is
   $O(n^2)$, so the rollout is subsampled to at most `optimal_transport.num_samples`
   snapshots (10,000, equal to the default coarse snapshot count, so the paper's setting
   uses every snapshot); the estimate is converged well below that, but its finite-sample
   bias means the value is only comparable at a fixed `num_samples`.
3. **Spatial autocorrelation RMSE**: relative RMSE of $\mathbb{E}[u(x)\overline{u(x+\Delta x)}]/\mathrm{Var}[u(x)]$ against the solver.
4. **Temporal autocorrelation RMSE**: relative RMSE of $\mathbb{E}[u(t)\overline{u(t+\Delta t)}]/\mathrm{Var}[u(x)]$ against the solver.
5. **Energy spectrum RMSE**: relative RMSE of $\mathbb{E}_{u}|\mathcal{F}_k(u)|^2$ against
   the solver.

Metrics 2 to 5 require an invariant measure and are computed for KS and CGLE only. They
draw on two cached rollouts: a coarse one, subsampled at `coarse_rollout.delta_t`, for
metrics 2, 3, and 5, and a fine one at the native `dt` for metric 4. Both discard
`burn_in` time units before recording. `long_horizon_rollout` is an extra diagnostic, the
rollout error against the solver as a function of time; it is not one of the five reported
metrics.

**KS**:
```bash
python scripts/eval.py \
    --checkpoint ./checkpoints/wasp_ks/model_step_100000.pt \
    --test_data_path ./data/ks_datasets/nu_1e-02_0.128t@128_5.0x@256.pt \
    --coarse_rollout.delta_t 0.5 \
    --coarse_rollout.burn_in 10 \
    --fine_rollout.burn_in 10 \
    --optimal_transport.eps 0.01 \
    --long_horizon_rollout False
```

**AC** (metric 1 only):
```bash
python scripts/eval.py \
    --checkpoint ./checkpoints/wasp_ac/model_step_20000.pt \
    --test_data_path ./data/ac_datasets/nu_1e-02_amp_1e-03_decay_1.0e+00_5.0t@128_6.283185307179586x@128.pt \
    --attractor_dist False \
    --energy_spectrum False \
    --spatial_correlation False \
    --temporal_correlation False \
    --long_horizon_rollout False
```

**CGLE** (a longer burn-in and tighter Sinkhorn regularization):
```bash
python scripts/eval.py \
    --checkpoint ./checkpoints/wasp_gl/model_step_200000.pt \
    --test_data_path ./data/gl_datasets/alpha_2e+00_beta_-2e+00_6.4t@128_50.0x@256.pt \
    --coarse_rollout.delta_t 25 \
    --coarse_rollout.burn_in 500 \
    --fine_rollout.burn_in 500 \
    --optimal_transport.eps 0.001 \
    --long_horizon_rollout False
```

Metrics are written to `./logs/eval/{wandb_name}/metrics.json` and the rollouts are cached
next to them, so re-running skips the rollout.
