"""CLI for generating the Complex Ginzburg-Landau dataset.

Wraps :meth:`GLEquation.create_dataset` with a zencfg config class so all
parameters can be overridden from the command line via ``--key value``.
Output is written to ``<root_dir>/alpha_<alpha>_beta_<beta>_<T>t@<Nt>_<L>x@<Nx>.pt``.
"""

from zencfg import ConfigBase, make_config_from_cli

from wasp.equations import GLEquation


class Default(ConfigBase):
    """Default configuration for Complex Ginzburg-Landau dataset generation.

    Attributes:
        alpha: Dispersion coefficient alpha in (1 + i*alpha)*Lap(u).
        beta: Nonlinear coefficient beta in (1 + i*beta)*|u|^2*u.
        batch_size: Number of trajectories solved in parallel.
        n_samples: Total number of trajectories to generate.
        domain: Target ``(T, L)`` time-space domain.
        resolution: Target ``(N_t, N_x)`` grid resolution.
        solver_domain: Extended ``(T_solver, L)`` domain used for burn-in.
        solver_resolution: Solver grid resolution before downsampling.
        device: Torch device string.
        seed: Random seed for reproducibility (None disables seeding).
        burn_in: Number of initial time steps discarded as burn-in.
        root_dir: Directory where the dataset file is saved.
        verbose: Whether to print progress.
    """
    alpha: float = 2.0
    beta: float = -2.0
    batch_size: int = 10
    n_samples: int = 10000
    domain: tuple[float, float] = (6.4, 50.0)
    resolution: tuple[int, int] = [128, 256]
    solver_domain: tuple[float, float] = (500.0, 50.0)
    solver_resolution: tuple[int, int] = [20000, 512]
    device: str = "cuda"
    seed: int | None = 0
    burn_in: int = 1000
    root_dir: str = "./data/gl_datasets"
    verbose: bool = True


if __name__ == "__main__":
    config = make_config_from_cli(Default)
    config = config.to_dict()
    GLEquation.create_dataset(
        alpha=config.alpha,
        beta=config.beta,
        batch_size=config.batch_size,
        n_samples=config.n_samples,
        domain=config.domain,
        resolution=config.resolution,
        solver_domain=config.solver_domain,
        solver_resolution=config.solver_resolution,
        device=config.device,
        seed=config.seed,
        burn_in=config.burn_in,
        root_dir=config.root_dir,
        verbose=config.verbose,
    )
