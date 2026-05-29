"""Post-training evaluation suite for WaSP models.

Measures solution accuracy and physical-observable fidelity via five metrics:
solution relative L2, invariant-measure W2 distance, spatial autocorrelation RMSE,
temporal autocorrelation RMSE, and energy-spectrum RMSE. Metrics 2-5 require
an attractor and are computed only for KS and CGLE.
"""

import json
import os

import torch
from tqdm import trange
from zencfg import ConfigBase, make_config_from_cli

from wasp.equations import EqSolution
from wasp.training import Trainer
from wasp.utils import OTMapping, l2_norm


class OptimalTransportConfig(ConfigBase):
    """Sinkhorn OT solver settings for the invariant-measure distance (metric 2).

    Attributes:
        num_samples: Number of samples used to estimate the W2 distance. The
            Sinkhorn coupling is O(n^2), so the rollout is subsampled to this
            size; the estimate is already converged well below it.
        eps: Entropy regularization strength.
        max_iters: Maximum Sinkhorn iterations.
        stop_thresh: Convergence threshold on the max dual-variable change.
    """

    num_samples: int = 10_000
    eps: float = 1e-3
    max_iters: int = 1000
    stop_thresh: float = 1e-5


class CoarseRolloutConfig(ConfigBase):
    """Settings for the coarse autoregressive rollout used for metrics 2, 3, and 5.

    Attributes:
        snapshots: Total number of snapshots to collect across all trajectories.
        n_traj: Number of independent trajectories to roll out in parallel.
        burn_in: Simulated time units to discard before recording.
        delta_t: Physical time between recorded snapshots.
    """

    snapshots: int = 10_000
    n_traj: int = 10
    burn_in: float = 10
    delta_t: float = 0.5


class FineRolloutConfig(ConfigBase):
    """Settings for the fine autoregressive rollout used for metric 4.

    Snapshots are recorded at the native step ``dt``, which the temporal
    autocorrelation needs.

    Attributes:
        snapshots: Total number of snapshots to collect across all trajectories.
        n_traj: Number of independent trajectories to roll out in parallel.
        burn_in: Simulated time units to discard before recording.
    """

    snapshots: int = 20_000
    n_traj: int = 20
    burn_in: float = 10


class LongRolloutConfig(ConfigBase):
    """Settings for the autoregressive rollout used in the long-horizon metric.

    Attributes:
        snapshots: Total number of snapshots to collect across all trajectories.
        n_traj: Number of independent trajectories to roll out in parallel.
    """

    snapshots: int = 10_000
    n_traj: int = 20


class EvalConfig(ConfigBase):
    """Top-level evaluation configuration.

    Attributes:
        checkpoint: Path to the ``.pt`` checkpoint file.
        test_data_path: Path to the test dataset ``.pt`` file.
        output_dir: Root directory for metric JSON files.
        coarse_rollout: Settings for the coarse rollout (metrics 2, 3, and 5).
        fine_rollout: Settings for the fine rollout (temporal autocorrelation).
        long_rollout: Settings for the rollout used by the long-horizon metric.
        optimal_transport: Sinkhorn solver settings for the W2 metric.
        test_set: Whether to evaluate solution L2 on the test set.
        attractor_dist: Whether to compute the invariant-measure distance (metric 2).
        energy_spectrum: Whether to compute the energy-spectrum RMSE.
        spatial_correlation: Whether to compute the spatial-autocorrelation RMSE.
        temporal_correlation: Whether to compute the temporal-autocorrelation RMSE.
        long_horizon_rollout: Whether to compute the long-horizon rollout error.
            This diagnostic is not among the five reported metrics.
    """

    checkpoint: str
    test_data_path: str
    output_dir: str = "./logs/eval"
    coarse_rollout: CoarseRolloutConfig = CoarseRolloutConfig()
    fine_rollout: FineRolloutConfig = FineRolloutConfig()
    long_rollout: LongRolloutConfig = LongRolloutConfig()
    optimal_transport: OptimalTransportConfig = OptimalTransportConfig()
    test_set: bool = True
    attractor_dist: bool = True
    energy_spectrum: bool = True
    spatial_correlation: bool = True
    temporal_correlation: bool = True
    long_horizon_rollout: bool = True


@torch.no_grad()
def test_set_eval(trainer: Trainer, test_sol: EqSolution) -> tuple[float, float, float]:
    """Evaluate the solution error and the PDE residual on the test set.

    Args:
        trainer: Fully set-up Trainer whose model is in eval mode.
        test_sol: Ground-truth solution trajectories from the test dataset.

    Returns:
        Tuple of (sol_abs_l2, sol_rel_l2, cond_abs_l2) over the test set. The
        residual cond_abs_l2 covers the forcing component f only; the
        initial-condition residual is not included.

    """
    batch_size = trainer.config.trainer.batch_size_eval
    condition = trainer.equation.map_to_test_condition(test_sol)
    sol_abs_l2 = 0.0
    sol_gt_l2 = 0.0
    cond_abs_l2 = 0.0
    for start_idx in range(0, test_sol.n_samples, batch_size):
        end_idx = min(start_idx + batch_size, test_sol.n_samples)
        sol_true = test_sol[start_idx:end_idx].to(
            device=trainer.device, dtype=trainer.dtype
        )
        cond_true = condition[start_idx:end_idx].to(
            device=trainer.device, dtype=trainer.dtype
        )

        norm_cond = trainer.normalizer.normalize_cond(cond_true)
        pred_u = trainer.model(trainer.equation.prepare_input(norm_cond))
        pred_u = trainer.equation.postprocess(
            pred_u, **trainer.config.trainer.postprocess
        )
        sol_pred = sol_true.new_solution(pred_u)
        sol_pred = trainer.normalizer.denormalize_sol(sol_pred)

        cond_pred = trainer.equation.forward(sol_pred)

        sol_abs_l2 += l2_norm(sol_pred["u"] - sol_true["u"]).item()
        sol_gt_l2 += l2_norm(sol_true["u"]).item()
        cond_abs_l2 += l2_norm(cond_pred["f"] - cond_true["f"]).item()
    print(f"Solution absolute L2: {sol_abs_l2 / test_sol.n_samples:.4e}")
    print(f"Solution relative L2: {sol_abs_l2 / sol_gt_l2:.4e}")
    print(f"Condition absolute L2: {cond_abs_l2 / test_sol.n_samples:.4e}")
    return (
        sol_abs_l2 / test_sol.n_samples,
        sol_abs_l2 / sol_gt_l2,
        cond_abs_l2 / test_sol.n_samples,
    )


@torch.no_grad()
def autoregressive_rollout(
    trainer: Trainer,
    test_sol: EqSolution,
    n_traj: int,
    snapshots: int,
    burn_in: float = 0,
    delta_t: float | None = None,
    solver: bool = True,
    model: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run autoregressive rollouts with the model and/or the solver.

    Starting from random initial conditions drawn from ``test_sol``, advances the
    model and/or the ETD-RK4 solver autoregressively and collects ``snapshots``
    snapshots after the burn-in period.

    Args:
        trainer: Fully set-up Trainer with the model and equation.
        test_sol: Pool of ground-truth solutions from which initial conditions
            are sampled.
        n_traj: Number of parallel trajectories.
        snapshots: Total snapshots to collect (must be divisible by ``n_traj``).
        burn_in: Physical time units to simulate before recording begins.
        delta_t: Physical time between recorded snapshots; defaults to ``dt``.
        solver: Whether to collect solver rollouts.
        model: Whether to collect model rollouts.

    Returns:
        Tuple ``(model_rollout, solver_rollout)`` of shape
        ``(n_traj, snapshots // n_traj, N_x)`` each, or ``None`` when the
        corresponding flag is False.

    """
    tmax, nt = trainer.equation.domain[0], trainer.equation.resolution[0]
    dt = tmax / nt

    # set delta_t to dt if it is None
    if delta_t is None:
        delta_t = dt

    # check arguments
    assert snapshots % n_traj == 0, "Snapshots must be divisible by n_traj"
    assert (delta_t / dt).is_integer(), "Delta t must be a multiple of dt"
    assert (burn_in / dt).is_integer(), "Burn in must be a multiple of dt"

    # compute time steps
    save_every = int(delta_t / dt)
    n_burn = int(burn_in / dt)

    # evaluate
    trainer.model.eval()
    outputs_model = []
    outputs_solver = []
    with torch.no_grad():
        sol = test_sol[torch.randperm(test_sol.n_samples)[:n_traj]].to(
            device=trainer.device, dtype=trainer.dtype
        )
        cond = trainer.equation.forward(sol).to(dtype=trainer.dtype)
        cond = trainer.equation.map_condition(cond, s=1)
        last_ic_model = cond["ic"]
        last_ic_solver = cond["ic"]
        for i in trange(
            (snapshots * save_every // n_traj + n_burn - 1) // (nt - 1) + 1
        ):
            if model:
                cond = cond.new_condition({"f": cond["f"], "ic": last_ic_model})
                norm_cond = trainer.normalizer.normalize_cond(cond)
                pred_u = trainer.model(trainer.equation.prepare_input(norm_cond))
                pred_u = trainer.equation.postprocess(
                    pred_u, **trainer.config.trainer.postprocess
                )
                sol_pred_model = sol.new_solution(pred_u)
                sol_pred_model = trainer.normalizer.denormalize_sol(sol_pred_model)
                last_ic_model = sol_pred_model["u"][:, -1, :]
            if solver:
                cond = cond.new_condition({"f": cond["f"], "ic": last_ic_solver})
                sol_pred_solver = trainer.equation.backward(cond)
                last_ic_solver = sol_pred_solver["u"][:, -1, :]
            if i >= (n_burn // (nt - 1)):
                idx = slice((n_burn - i * (nt - 1)) % save_every, -1, save_every)
                if model:
                    outputs_model.append(sol_pred_model["u"][:, idx, :].cpu())
                if solver:
                    outputs_solver.append(sol_pred_solver["u"][:, idx, :].cpu())
    if model:
        outputs_model = torch.cat(outputs_model, dim=1)[: snapshots // n_traj]
    else:
        outputs_model = None
    if solver:
        outputs_solver = torch.cat(outputs_solver, dim=1)[: snapshots // n_traj]
    else:
        outputs_solver = None
    return outputs_model, outputs_solver


def collect_coarse_rollouts(
    trainer: Trainer,
    test_sol: EqSolution,
    config: CoarseRolloutConfig,
    save_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect coarse-grained rollouts for attractor and spectral metrics.

    Results are cached to ``<save_path>/coarse_rollouts.pt`` and reloaded on
    subsequent calls, so re-running eval skips expensive rollout computation.

    Args:
        trainer: Fully set-up Trainer with the model and equation.
        test_sol: Ground-truth solutions used to seed initial conditions.
        config: Coarse rollout hyperparameters (snapshots, burn-in, delta_t).
        save_path: Directory in which to cache the rollout tensors.

    Returns:
        Tuple ``(model_rollout, solver_rollout_ref, solver_rollout_pred)``
        each flattened to shape ``(snapshots, N_x)``.

    """
    if os.path.exists(save_path + "/coarse_rollouts.pt"):
        return torch.load(save_path + "/coarse_rollouts.pt")

    model_rollout, solver_rollout_pred = autoregressive_rollout(
        trainer,
        test_sol,
        n_traj=config.n_traj,
        snapshots=config.snapshots,
        burn_in=config.burn_in,
        delta_t=config.delta_t,
    )

    _, solver_rollout_ref = autoregressive_rollout(
        trainer,
        test_sol,
        n_traj=config.n_traj,
        snapshots=config.snapshots,
        burn_in=config.burn_in,
        delta_t=config.delta_t,
        solver=True,
        model=False,
    )

    torch.save(
        (
            model_rollout.flatten(0, 1),
            solver_rollout_ref.flatten(0, 1),
            solver_rollout_pred.flatten(0, 1),
        ),
        save_path + "/coarse_rollouts.pt",
    )

    return (
        model_rollout.flatten(0, 1),
        solver_rollout_ref.flatten(0, 1),
        solver_rollout_pred.flatten(0, 1),
    )


def collect_fine_rollouts(
    trainer: Trainer,
    test_sol: EqSolution,
    config: FineRolloutConfig,
    save_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect rollouts at the native dt for the temporal autocorrelation.

    Results are cached to ``<save_path>/fine_rollouts.pt`` and reloaded on
    subsequent calls.

    Args:
        trainer: Fully set-up Trainer with the model and equation.
        test_sol: Ground-truth solutions used to seed initial conditions.
        config: Fine rollout hyperparameters (snapshots, n_traj, burn_in).
        save_path: Directory in which to cache the rollout tensors.

    Returns:
        Tuple ``(model_rollout, solver_rollout)`` each of shape
        ``(n_traj, snapshots // n_traj, N_x)``.

    """
    if os.path.exists(save_path + "/fine_rollouts.pt"):
        return torch.load(save_path + "/fine_rollouts.pt")

    model_rollout, solver_rollout = autoregressive_rollout(
        trainer,
        test_sol,
        n_traj=config.n_traj,
        snapshots=config.snapshots,
        burn_in=config.burn_in,
    )

    torch.save(
        (model_rollout, solver_rollout),
        save_path + "/fine_rollouts.pt",
    )

    return model_rollout, solver_rollout


def collect_long_rollouts(
    trainer: Trainer,
    test_sol: EqSolution,
    config: LongRolloutConfig,
    save_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect paired model and solver rollouts for the long-horizon metric.

    Results are cached to ``<save_path>/long_rollouts.pt`` and reloaded on
    subsequent calls.

    Args:
        trainer: Fully set-up Trainer with the model and equation.
        test_sol: Ground-truth solutions used to seed initial conditions.
        config: Long rollout hyperparameters (snapshots, n_traj).
        save_path: Directory in which to cache the rollout tensors.

    Returns:
        Tuple ``(model_rollout, solver_rollout)`` each of shape
        ``(n_traj, snapshots // n_traj, N_x)``.

    """
    if os.path.exists(save_path + "/long_rollouts.pt"):
        return torch.load(save_path + "/long_rollouts.pt")

    model_rollout, solver_rollout = autoregressive_rollout(
        trainer,
        test_sol,
        n_traj=config.n_traj,
        snapshots=config.snapshots,
    )

    torch.save(
        (model_rollout, solver_rollout),
        save_path + "/long_rollouts.pt",
    )

    return model_rollout, solver_rollout


def attractor_orbit_eval(
    trainer: Trainer,
    model_rollout: torch.Tensor,
    solver_rollout_ref: torch.Tensor,
    solver_rollout_pred: torch.Tensor,
    ot_config: OptimalTransportConfig,
) -> tuple[float, float, float, float]:
    """Compute the invariant-measure distance (Sinkhorn W2) to the solver reference.

    Args:
        trainer: Trainer providing device and equation metadata.
        model_rollout: Flattened model snapshots of shape ``(snapshots, N_x)``.
        solver_rollout_ref: Independent reference solver snapshots used as target.
        solver_rollout_pred: Paired solver snapshots (same ICs as model) used to
            estimate the solver-only reference distance.
        ot_config: Sinkhorn solver settings.

    Returns:
        Tuple ``(model_dist, solver_dist, model_dist_se, solver_dist_se)`` where
        each distance is the Sinkhorn W2 approximation and ``_se`` is its standard
        error.

    """
    ot_mapping = OTMapping(
        device=trainer.device,
        eps=ot_config.eps,
        max_iters=ot_config.max_iters,
        stop_thresh=ot_config.stop_thresh,
        verbose=True,
    )

    if model_rollout.dtype.is_complex:
        model_rollout = torch.view_as_real(model_rollout).flatten(1, 2)
    if solver_rollout_ref.dtype.is_complex:
        solver_rollout_ref = torch.view_as_real(solver_rollout_ref).flatten(1, 2)
    if solver_rollout_pred.dtype.is_complex:
        solver_rollout_pred = torch.view_as_real(solver_rollout_pred).flatten(1, 2)

    if model_rollout.shape[0] > ot_config.num_samples:
        model_rollout = model_rollout[
            torch.randperm(model_rollout.shape[0])[: ot_config.num_samples]
        ]
        solver_rollout_ref = solver_rollout_ref[
            torch.randperm(solver_rollout_ref.shape[0])[: ot_config.num_samples]
        ]
        solver_rollout_pred = solver_rollout_pred[
            torch.randperm(solver_rollout_pred.shape[0])[: ot_config.num_samples]
        ]

    model_dist, model_dist_se = ot_mapping.fit(
        model_rollout, solver_rollout_ref, return_se=True
    )
    solver_dist, solver_dist_se = ot_mapping.fit(
        solver_rollout_pred, solver_rollout_ref, return_se=True
    )
    print(f"ML OT distance: {model_dist:.4e} +/- {model_dist_se:.4e}")
    print(f"Solver OT distance: {solver_dist:.4e} +/- {solver_dist_se:.4e}")

    return model_dist, solver_dist, model_dist_se, solver_dist_se


def energy_spectrum_eval(
    trainer: Trainer,
    model_rollout: torch.Tensor,
    solver_rollout: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    """Compare the time-averaged energy spectra of the model and solver rollouts.

    Args:
        trainer: Trainer providing equation domain/resolution and dtype.
        model_rollout: Model snapshots of shape ``(snapshots, N_x)``.
        solver_rollout: Solver snapshots of shape ``(snapshots, N_x)``.

    Returns:
        Tuple ``(model_spectrum, solver_spectrum, freqs)`` where each element is a
        list of floats corresponding to Fourier modes.

    """
    if trainer.dtype.is_complex:
        fft_func = torch.fft.fft
        fftfreq_func = torch.fft.fftfreq
    else:
        fft_func = torch.fft.rfft
        fftfreq_func = torch.fft.rfftfreq

    model_spectrum = (
        fft_func(model_rollout, norm="forward", dim=-1).abs().square().mean(dim=0)
    )
    solver_spectrum = (
        fft_func(solver_rollout, norm="forward", dim=-1).abs().square().mean(dim=0)
    )

    rel_error = torch.sqrt(
        torch.square(torch.abs(model_spectrum - solver_spectrum)).sum()
        / torch.square(torch.abs(solver_spectrum)).sum()
    )
    print(f"Energy Spectrum Relative Error: {rel_error.item():.4e}")
    xmax, nx = trainer.equation.domain[1], trainer.equation.resolution[1]
    return (
        model_spectrum.tolist(),
        solver_spectrum.tolist(),
        fftfreq_func(nx, d=xmax / (2 * torch.pi * nx)).tolist(),
    )


def spatial_correlation_eval(
    trainer: Trainer,
    model_rollout: torch.Tensor,
    solver_rollout: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    """Compute and compare spatial autocorrelation functions for model and solver.

    Args:
        trainer: Trainer providing equation domain/resolution.
        model_rollout: Model snapshots of shape ``(snapshots, N_x)``.
        solver_rollout: Solver snapshots of shape ``(snapshots, N_x)``.

    Returns:
        Tuple ``(model_correlation, solver_correlation, shift_amount)`` each a list
        of floats. Correlations are normalized to 1 at zero lag.

    """
    model_rollout = model_rollout - model_rollout.mean()
    solver_rollout = solver_rollout - solver_rollout.mean()

    xmax, nx = trainer.equation.domain[1], trainer.equation.resolution[1]
    shift_amount = torch.linspace(0, xmax, nx + 1)[: nx // 2 + 1]
    model_correlation = torch.zeros(nx // 2 + 1, dtype=trainer.dtype)
    solver_correlation = torch.zeros(nx // 2 + 1, dtype=trainer.dtype)
    for shift in range(nx // 2 + 1):
        model_rollout_shifted = torch.roll(model_rollout, shift, dims=-1)
        solver_rollout_shifted = torch.roll(solver_rollout, shift, dims=-1)
        model_correlation[shift] = (model_rollout * model_rollout_shifted.conj()).mean()
        solver_correlation[shift] = (
            solver_rollout * solver_rollout_shifted.conj()
        ).mean()

    model_correlation = model_correlation / model_correlation[0]
    solver_correlation = solver_correlation / solver_correlation[0]

    rel_error = torch.sqrt(
        torch.square(torch.abs(model_correlation - solver_correlation)).sum()
        / solver_correlation.abs().square().sum()
    )

    print(f"Spatial Correlation Relative Error: {rel_error.item():.4e}")
    return (
        model_correlation.tolist(),
        solver_correlation.tolist(),
        shift_amount.tolist(),
    )


def temporal_correlation_eval(
    trainer: Trainer,
    model_rollout: torch.Tensor,
    solver_rollout: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    """Compute and compare temporal autocorrelation functions for model and solver.

    Args:
        trainer: Trainer providing equation domain/resolution.
        model_rollout: Model snapshots of shape ``(n_traj, snapshots, N_x)``.
        solver_rollout: Solver snapshots of shape ``(n_traj, snapshots, N_x)``.

    Returns:
        Tuple ``(model_correlation, solver_correlation, shift_amount)`` each a list
        of floats, evaluated at lags 0 to ``N_t - 1`` in units of the native
        ``dt``. Correlations are normalized to 1 at zero lag.

    """
    model_rollout = model_rollout - model_rollout.mean()
    solver_rollout = solver_rollout - solver_rollout.mean()

    tmax, nt = trainer.equation.domain[0], trainer.equation.resolution[0]
    shift_amount = torch.linspace(0, tmax, nt + 1)[:-1]
    model_correlation = torch.zeros(nt, dtype=trainer.dtype)
    solver_correlation = torch.zeros(nt, dtype=trainer.dtype)
    model_correlation[0] = model_rollout.abs().square().mean()
    solver_correlation[0] = solver_rollout.abs().square().mean()
    for shift in range(1, nt):
        model_correlation[shift] = (
            model_rollout[:, :-shift] * model_rollout[:, shift:].conj()
        ).mean()
        solver_correlation[shift] = (
            solver_rollout[:, :-shift] * solver_rollout[:, shift:].conj()
        ).mean()

    model_correlation = model_correlation / model_correlation[0]
    solver_correlation = solver_correlation / solver_correlation[0]

    rel_error = torch.sqrt(
        torch.square(torch.abs(model_correlation - solver_correlation)).sum()
        / solver_correlation.abs().square().sum()
    )

    print(f"Temporal Correlation Relative Error: {rel_error.item():.4e}")
    return (
        model_correlation.tolist(),
        solver_correlation.tolist(),
        shift_amount.tolist(),
    )


def long_horizon_rollout_eval(
    trainer: Trainer,
    model_rollout: torch.Tensor,
    solver_rollout: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Compute per-time-step RMS error of model rollout relative to solver.

    Args:
        trainer: Trainer providing equation domain/resolution.
        model_rollout: Model trajectory of shape ``(n_traj, T, N_x)``.
        solver_rollout: Solver trajectory of shape ``(n_traj, T, N_x)``.

    Returns:
        Tuple ``(error, time_step)`` where ``error[i]`` is the mean RMS error at
        step ``i`` and ``time_step[i]`` is the corresponding physical time.

    """
    tmax, nt = trainer.equation.domain[0], trainer.equation.resolution[0]
    dt = tmax / nt
    time_step = torch.linspace(
        0, (model_rollout.shape[1] - 1) * dt, model_rollout.shape[1]
    )
    error = torch.linalg.norm(model_rollout - solver_rollout, dim=-1) / (
        model_rollout.shape[-1] ** 0.5
    )
    error = error.mean(dim=0)
    print(f"Long Horizon Rollout Error: {error.mean().item():.4e}")
    return error.tolist(), time_step.tolist()


def main():
    """Parse CLI config, run the selected evaluation metrics, and write metrics.json."""
    config = make_config_from_cli(EvalConfig)
    config: EvalConfig = config.to_dict()
    trainer = Trainer.load_checkpoint(config.checkpoint)
    trainer.model.eval()
    test_sol = EqSolution.load(config.test_data_path)
    output_path = os.path.join(config.output_dir, f"{trainer.config.wandb.name}")
    os.makedirs(output_path, exist_ok=True)
    output = {}
    if config.test_set:
        sol_abs_l2, sol_rel_l2, cond_abs_l2 = test_set_eval(trainer, test_sol)
        output["sol_abs_l2"] = sol_abs_l2
        output["sol_rel_l2"] = sol_rel_l2
        output["cond_abs_l2"] = cond_abs_l2
    if any(
        [
            config.attractor_dist,
            config.energy_spectrum,
            config.spatial_correlation,
        ]
    ):
        model_rollout, solver_rollout_ref, solver_rollout_pred = (
            collect_coarse_rollouts(
                trainer, test_sol, config.coarse_rollout, output_path
            )
        )
    if config.attractor_dist:
        model_dist, solver_dist, model_dist_se, solver_dist_se = attractor_orbit_eval(
            trainer,
            model_rollout=model_rollout,
            solver_rollout_ref=solver_rollout_ref,
            solver_rollout_pred=solver_rollout_pred,
            ot_config=config.optimal_transport,
        )
        output["model_dist"] = model_dist
        output["solver_dist"] = solver_dist
        output["model_dist_se"] = model_dist_se
        output["solver_dist_se"] = solver_dist_se
    if config.energy_spectrum:
        model_spectrum, solver_spectrum, energy_freqs = energy_spectrum_eval(
            trainer,
            model_rollout=model_rollout,
            solver_rollout=solver_rollout_pred,
        )
        output["model_spectrum"] = model_spectrum
        output["solver_spectrum"] = solver_spectrum
        output["energy_freqs"] = energy_freqs
    if config.spatial_correlation:
        model_correlation, solver_correlation, shift_amount = spatial_correlation_eval(
            trainer,
            model_rollout=model_rollout,
            solver_rollout=solver_rollout_pred,
        )
        output["model_spatial_correlation"] = model_correlation
        output["solver_spatial_correlation"] = solver_correlation
        output["spatial_shift_amount"] = shift_amount

    if config.temporal_correlation:
        model_rollout, solver_rollout = collect_fine_rollouts(
            trainer, test_sol, config.fine_rollout, output_path
        )

        model_correlation, solver_correlation, shift_amount = temporal_correlation_eval(
            trainer,
            model_rollout=model_rollout,
            solver_rollout=solver_rollout,
        )
        output["model_temporal_correlation"] = model_correlation
        output["solver_temporal_correlation"] = solver_correlation
        output["temporal_shift_amount"] = shift_amount

    if config.long_horizon_rollout:
        model_rollout, solver_rollout = collect_long_rollouts(
            trainer, test_sol, config.long_rollout, output_path
        )
        error, time_step = long_horizon_rollout_eval(
            trainer,
            model_rollout=model_rollout,
            solver_rollout=solver_rollout,
        )
        output["long_horizon_rollout_error"] = error
        output["long_horizon_rollout_time_step"] = time_step

    def complex_encoder(obj):
        """Encode Python complex numbers as ``["complex", real, imag]`` for JSON."""
        if isinstance(obj, complex):
            return ["complex", obj.real, obj.imag]
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(output_path + "/metrics.json", "w") as f:
        json.dump(output, f, indent=2, default=complex_encoder)
    print(f"Saved output to {output_path}/metrics.json")


if __name__ == "__main__":
    main()
