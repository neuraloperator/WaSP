"""CLI for generating the Kuramoto-Sivashinsky dataset.

Wraps :meth:`KSEquation.create_dataset` with a zencfg config class so all
parameters can be overridden from the command line via ``--key value``.
Output is written to ``<root_dir>/nu_<nu>_<T>t@<Nt>_<L>x@<Nx>.pt``.
"""

from zencfg import ConfigBase, make_config_from_cli

from wasp.equations import KSEquation


class Default(ConfigBase):
    """Default configuration for KS dataset generation.

    Attributes:
        nu: Hyperviscosity coefficient.
        batch_size: Number of trajectories solved in parallel.
        n_samples: Total number of trajectories to generate.
        domain: Target ``(T, L)`` time-space domain.
        resolution: Target ``(N_t, N_x)`` grid resolution.
        solver_domain: Extended ``(T_solver, L)`` domain used for burn-in.
        solver_resolution: Solver grid resolution before downsampling.
        device: Torch device string.
        seed: Random seed for reproducibility.
        burn_in: Number of initial time steps discarded as burn-in.
        root_dir: Directory where the dataset file is saved.
        verbose: Whether to print progress.
    """
    nu: float = 0.01
    batch_size: int = 10
    n_samples: int = 10000
    domain: tuple[float, float] = (0.128, 5)
    resolution: tuple[int, int] = (128, 256)
    solver_domain: tuple[float, float] = (10, 5)
    solver_resolution: tuple[int, int] = (20000, 512)
    device: str = "cuda"
    seed: int = 0
    burn_in: int = 1000
    root_dir: str = "./data/ks_datasets"
    verbose: bool = True


if __name__ == "__main__":
    config = make_config_from_cli(Default)
    config = config.to_dict()
    KSEquation.create_dataset(
        nu=config.nu,
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
