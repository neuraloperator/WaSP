"""CLI for generating the Allen-Cahn dataset.

Wraps :meth:`ACEquation.create_dataset` with a zencfg config class so all
parameters can be overridden from the command line via ``--key value``.
Output is written to ``<root_dir>/nu_<nu>_amp_<amp>_decay_<decay>_<T>t@<Nt>_<L>x@<Nx>.pt``.
"""

import math

from zencfg import ConfigBase, make_config_from_cli

from wasp.equations import ACEquation


class Default(ConfigBase):
    """Default configuration for Allen-Cahn dataset generation.

    Attributes:
        nu: Diffusion coefficient.
        ic_amplitude: Amplitude of the Gaussian random field used for initial conditions.
        power_decay: Spectral decay exponent (P_k proportional to exp(-k * power_decay)).
        batch_size: Number of trajectories solved in parallel.
        n_samples: Total number of trajectories to generate.
        domain: Target ``(T, L)`` time-space domain.
        resolution: Target ``(N_t, N_x)`` grid resolution.
        solver_resolution: Solver grid resolution before downsampling.
        device: Torch device string.
        seed: Random seed for reproducibility.
        root_dir: Directory where the dataset file is saved.
    """
    nu: float = 0.01
    ic_amplitude: float = 1e-3
    power_decay: float = 1.0
    batch_size: int = 10
    n_samples: int = 1000
    domain: tuple[float, float] = (5.0, 2 * math.pi)
    resolution: tuple[int, int] = (128, 128)
    solver_resolution: tuple[int, int] = (256, 256)
    device: str = "cuda"
    seed: int = 0
    root_dir: str = "./data/ac_datasets"


if __name__ == "__main__":
    config = make_config_from_cli(Default)
    config = config.to_dict()
    ACEquation.create_dataset(
        nu=config.nu,
        ic_amplitude=config.ic_amplitude,
        power_decay=config.power_decay,
        batch_size=config.batch_size,
        n_samples=config.n_samples,
        domain=config.domain,
        resolution=config.resolution,
        solver_resolution=config.solver_resolution,
        device=config.device,
        seed=config.seed,
        root_dir=config.root_dir,
    )
