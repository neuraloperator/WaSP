import os

import numpy as np
import torch
from matplotlib import colors
from torch import fft
from tqdm import trange

from ..utils import FiniteDifference, OTMapping, h1_norm, l2_norm, sample_grf
from .equation_base import EquationBase
from .equation_data import EqCondition, EqData, EqSolution


class GLEquation(EquationBase):
    """
    The complex Ginzburg-Landau (CGLE) equation with an added forcing term:
        u_t = u + (1 + i*alpha)Lap(u) - (1 + i*beta)|u|^2u + f(t, x)
    where f is the forcing field. The input is v = (u(0, x), f(t, x)), stored
    as an EqCondition, which makes the PDE operator N a bijection.
    """

    ot_map: OTMapping = None
    ic_power_spectrum: torch.Tensor = None
    test_data: dict[str, EqData] = {}
    solution_normalizer_key: list[str] = ["u"]
    condition_normalizer_key: list[str] = ["f", "ic"]

    def __init__(
        self,
        root_dir: str,
        alpha: float,
        beta: float,
        domain: tuple[float, float],
        resolution: tuple[int, int],
        loss_weights: dict[str, float],
        ot_rank: int,
        fitting_samples: int,
        test_samples: int,
        ot_eps: float = 1e-2,
        ot_max_iters: int = 1000,
        ot_stop_thresh: float = 1e-5,
        verbose: bool = False,
    ) -> None:

        super().__init__(
            domain=domain,
            resolution=resolution,
        )

        self.solution_shapes = {"u": resolution}
        self.condition_shapes = {"f": resolution, "ic": (resolution[1],)}

        self.root_dir = root_dir
        self.alpha = alpha
        self.beta = beta
        self.loss_weights = loss_weights
        self.ot_rank = ot_rank
        self.fitting_samples = fitting_samples
        self.test_samples = test_samples
        self.ot_eps = ot_eps
        self.ot_max_iters = ot_max_iters
        self.ot_stop_thresh = ot_stop_thresh
        self.verbose = verbose

        self.finite_diff = FiniteDifference(N=5, degree=4)

    def setup(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Load the test dataset, fit the space-time power spectrum of the
        manufactured perturbation and the plane-wave wavenumber distribution, and
        fit the transport map to the target initial-condition distribution.

        The spatial factor of the spectrum is estimated from the target initial
        conditions; the Gaussian temporal factor is calibrated to the time
        derivative the unforced equation gives at t = 0.

        Args:
            device: Device to store the fitted statistics and mappings

        """
        assert os.path.exists(self.root_dir), (
            f"Root directory {self.root_dir} does not exist."
        )

        if self.verbose:
            print("Setting up GL equation...")
            print("Loading test datasets...")

        # load the solver dataset; one split is held out for testing and another
        # fits the power spectrum and the transport map
        filename = os.path.join(
            self.root_dir,
            f"alpha_{self.alpha:.0e}_beta_{self.beta:.0e}_{self.domain[0]}t@"
            f"{self.resolution[0]}_{self.domain[1]}x@{self.resolution[1]}.pt",
        )
        assert os.path.exists(filename), f"Test dataset file {filename} does not exist."
        eq_solution = EqSolution.load(filename)

        # split the dataset into test and fitting
        sample_indices = torch.randperm(eq_solution.n_samples)
        self.test_data = {"solution": eq_solution[sample_indices[: self.test_samples]]}

        # fit the power spectrum and optimal transport mapping
        fitting_solution = eq_solution[
            sample_indices[self.test_samples : self.test_samples + self.fitting_samples]
        ]
        fitting_condition = self.forward(fitting_solution)

        # cache the power spectrum of initial condition
        ic = fitting_condition["ic"]
        ic_hat = fft.fft(ic, norm="forward")
        self.ic_power_spectrum = ic_hat.abs().square().mean(dim=0)

        # extract parameters
        nt, nx = self.resolution
        tmax, xmax = self.domain

        # calculate temporal power by simple Gaussian estimate to match the time
        # derivative at t=0, u_t = u + (1 + i*alpha)Lap(u) - (1 + i*beta)|u|^2u
        kt = torch.fft.fftfreq(2 * nt, d=tmax / (2 * torch.pi * nt))
        kx = torch.fft.fftfreq(nx, d=xmax / (2 * torch.pi * nx))
        du_dt_hat = (
            ic_hat
            + (1 + 1j * self.alpha) * (-(kx**2)) * ic_hat
            - (1 + 1j * self.beta) * torch.fft.fft(ic.abs() ** 2 * ic, norm="forward")
        )
        kt_var = (
            du_dt_hat.abs().square().mean(dim=0).sum() / self.ic_power_spectrum.sum()
        )

        # compute temporal power spectrum and construct 2D power spectrum
        power_t = kt.square().div(2 * kt_var).neg().exp()
        power_t = power_t / power_t.sum()
        self.power_2d = torch.einsum("t,x->tx", power_t, self.ic_power_spectrum)

        # zero higher mode's ic spectrum
        self.ic_power_spectrum[kx.abs() > 0.8] = 0

        # fit optimal transport mapping
        if self.verbose:
            print("Fitting optimal transport mapping...")

        random_solutions = self.sample_solution(n_samples=self.fitting_samples)
        fitting_condition = fitting_condition.to(device=device, dtype=torch.complex64)
        ic_samples = fitting_condition["ic"]
        random_samples = random_solutions.to(dtype=torch.complex64)["u"][:, 0, :]

        self.ot_map = OTMapping(
            device=device,
            eps=self.ot_eps,
            max_iters=self.ot_max_iters,
            stop_thresh=self.ot_stop_thresh,
            q=self.ot_rank,
            verbose=self.verbose,
        )
        dist = self.ot_map.fit(
            torch.view_as_real(random_samples).flatten(1, -1),
            torch.view_as_real(ic_samples).flatten(1, -1),
        )
        if self.verbose:
            print(f"Fitted OT mapping with 2-Wasserstein distance {dist:.6f}")

    def forward(self, solution: EqSolution) -> EqCondition:
        """
        Apply the PDE operator to a solution, v = N(u).

        Args:
            solution: EqSolution object representing the solution field
        Returns:
            EqCondition holding the derived input v = (u(0, x), f(t, x))

        """
        self.check_solution(solution)

        # cast to complex128 for better accuracy in fft operations
        dtype = solution.dtype
        solution = solution.to(dtype=torch.complex128)

        # extract parameters
        nt, nx = self.resolution
        tmax, xmax = self.domain
        u = solution["u"]

        # setup spatial frequencies and time derivative module
        kx = fft.fftfreq(nx, d=xmax / (2 * torch.pi * nx)).to(u)

        # Fourier transform the solution (v holds fft(u) here, not the input v)
        v = fft.fft(u, norm="forward")

        # compute linear and nonlinear terms in Fourier space
        linear_term = -(1 - (1 + 1j * self.alpha) * kx.pow(2)) * v
        nonlinear_term = (1 + 1j * self.beta) * torch.fft.fft(
            u.abs().pow(2) * u, norm="forward"
        )

        # compute time derivative in regular space
        time_derivative = self.finite_diff.derive(u, delta=tmax / nt, dim=1)

        # transform spatial derivative back to regular space
        spatial_derivative = fft.ifft(linear_term + nonlinear_term, norm="forward")

        # compute forcing term f(t, x)
        f = time_derivative + spatial_derivative

        # return the input v as an EqCondition
        return EqCondition(
            {"f": f.to(dtype), "ic": u[:, 0, :].to(dtype)}, domain=self.domain
        )

    def backward(
        self,
        condition: EqCondition,
        M: int | None = 128,
    ) -> EqSolution:
        """
        Solve the equation for u = G(v) given the input v. This solver is
        based on the ETDRK4 scheme for time integration.

        Args:
            condition: EqCondition holding the input v
            M: the number of integration points for the contour integral
        Returns:
            EqSolution object representing the solved solution field

        """
        self.check_condition(condition)

        # extract parameters
        nt, nx = self.resolution
        tmax, xmax = self.domain
        device = condition.device
        dt = tmax / nt

        # cast to complex128
        dtype = condition.dtype
        condition = condition.to(dtype=torch.complex128)

        # setup spatial frequencies
        kx = fft.fftfreq(nx, d=xmax / (2 * torch.pi * nx)).to(device)
        dealias_mask = kx.abs() > 1 / 3 * (2 * torch.pi * nx) / xmax

        # Compute and cache ETDRK4 coefficients
        if not hasattr(self, "_etdrk4_cache"):
            self._etdrk4_L = 1 - (1 + 1j * self.alpha) * kx.pow(2)
            self._etdrk4_dt = dt
            self._etdrk4_M = M
        E, E_2, q, f1, f2, f3 = self._solve_coeff(device, is_complex=True)

        # define stepping functions
        def nonlinear_term_spectral(hat_u: torch.Tensor) -> torch.Tensor:
            u = fft.ifft(hat_u, norm="forward")
            nonlinear_term = -(1 + 1j * self.beta) * u.abs().pow(2) * u
            return fft.fft(nonlinear_term, norm="forward")

        def linear_term_exponential_time(hat_u: torch.Tensor) -> torch.Tensor:
            return E * hat_u

        def dealias(hat_u: torch.Tensor) -> torch.Tensor:
            hat_u[..., dealias_mask] = 0
            return hat_u

        # time-stepping loop
        u = condition["ic"]
        u_history = torch.empty((u.size(0), nt, nx), device=device, dtype=u.dtype)
        u_history[:, 0, :] = u

        # Transform initial condition and forcing term to Fourier space
        # upsample f to match ETDRK4 time steps
        f_fft = fft.fft(condition["f"], norm="forward")
        v = fft.fft(u, norm="forward")

        for n in range(1, nt):
            # compute f_fft at different time:
            f_fft_curr = f_fft[:, n - 1]
            f_fft_mid = 0.5 * f_fft[:, n - 1] + 0.5 * f_fft[:, n]
            f_fft_next = f_fft[:, n]

            # Compute nonlinear terms with ETDRK4
            nl_v = nonlinear_term_spectral(v) + f_fft_curr
            a = E_2 * v + q * nl_v
            nl_a = nonlinear_term_spectral(a) + f_fft_mid
            b = E_2 * v + q * nl_a
            nl_b = nonlinear_term_spectral(b) + f_fft_mid
            c = E_2 * a + q * (2 * nl_b - nl_v)
            nl_c = nonlinear_term_spectral(c) + f_fft_next
            nonlinear_term = f1 * nl_v + 2 * f2 * (nl_b + nl_a) + f3 * nl_c

            # Compute linear term with exponential time operator
            linear_term = linear_term_exponential_time(v)

            # Update the solution using ETDRK4 method
            v = linear_term + nonlinear_term

            # Dealias the solution
            v = dealias(v)

            # Store solution at current time step
            u_history[:, n, :] = fft.ifft(v, norm="forward")

        return EqSolution({"u": u_history.to(dtype=dtype)}, domain=self.domain)

    def sample_solution(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> EqSolution:
        """
        Draw solutions from the manufactured distribution mu_0: an unstable plane
        wave whose wavenumber is drawn with probability proportional to the target
        initial-condition power spectrum and whose frequency follows the CGLE
        dispersion relation, modulated by a complex Gaussian random field. The
        field is sampled on a doubled temporal grid and the first half is kept,
        which removes the periodicity an FFT-sampled field would impose along the
        time axis.

        Args:
            n_samples: Number of samples to generate
            device: Device to store the sampled solutions
        Returns:
            EqSolution object representing the sampled solutions

        """
        assert self.power_2d is not None, (
            "Initial condition power spectrum not computed. "
            "Please run setup() before sampling solutions."
        )

        nt, nx = self.resolution
        tmax, xmax = self.domain

        # sample from Gaussian random field with the calculated power spectrum
        u_random = sample_grf(
            self.power_2d, n_samples, (2 * nt, nx), device=device, complex_valued=True
        )

        # Sample plane wave solutions
        kx = torch.fft.fftfreq(
            nx, d=xmax / (2 * torch.pi * nx), dtype=torch.float64, device=device
        )
        index = torch.multinomial(
            self.ic_power_spectrum, num_samples=n_samples, replacement=True
        ).view(-1, 1, 1)
        omega = -self.beta - (self.alpha - self.beta) * kx[index] ** 2
        # setup grid
        T, X = torch.meshgrid(
            torch.linspace(0, tmax, nt + 1, dtype=torch.float64, device=device)[:-1],
            torch.linspace(0, xmax, nx + 1, dtype=torch.float64, device=device)[:-1],
            indexing="ij",
        )
        # Generate plane-wave solution
        u_wave = torch.exp(1j * omega * T[None] + 1j * kx[index] * X[None])

        # combine the two
        u = (0.5**0.5) * (1 + u_random[:, :nt, :]) * u_wave

        # return the sampled solutions
        return EqSolution({"u": u}, domain=self.domain)

    def map_condition(self, condition: EqCondition, s: float) -> EqCondition:
        """
        Move the inputs along the McCann interpolation
        nu_s = ((1 - s) Id + s T)_# nu_0: the initial condition is interpolated
        toward its transport image T(ic), and the forcing anneals as (1 - s) f.

        Args:
            condition: EqCondition holding the inputs v
            s: interpolation parameter of the transport path, in [0, 1]

        Returns:
            EqCondition holding the interpolated inputs, distributed as nu_s

        """
        self.check_condition(condition)
        assert self.ot_map is not None, (
            "Optimal transport mapping not computed. "
            "Please run setup() before mapping conditions."
        )
        assert 0.0 <= s <= 1.0, "s must be in the range [0, 1]."
        ic_real = torch.view_as_real(condition["ic"]).flatten(1, -1)
        mapped_ic_real = self.ot_map.map(ic_real).view(condition.n_samples, -1, 2)
        mapped_ic = torch.view_as_complex(mapped_ic_real)
        return condition.new_condition(
            {
                "f": (1 - s) * condition["f"],
                "ic": (1 - s) * condition["ic"] + s * mapped_ic,
            }
        )

    def map_to_test_condition(self, solution: EqSolution) -> EqCondition:
        """
        Build the target input of a solution: its initial condition, zero forcing.

        Args:
            solution: EqSolution of solver trajectories
        Returns:
            EqCondition holding the target inputs v = (u(0, x), 0)

        """
        self.check_solution(solution)
        return EqCondition(
            {
                "f": torch.zeros_like(solution["u"]),
                "ic": solution["u"][:, 0, :],
            },
            domain=solution.domain,
        )

    def prepare_input(self, condition: EqCondition) -> torch.Tensor:
        """
        Prepare the input for the neural network.

        Args:
            condition: EqCondition holding the inputs v
        Returns:
            torch.Tensor representing the prepared input

        """
        self.check_condition(condition)

        # continue the domain
        nt, nx = self.resolution

        inputs = [
            torch.view_as_real(condition["f"].view(-1, nt, nx)).permute(0, 3, 1, 2)
        ]
        inputs.append(
            torch.view_as_real(
                condition["ic"].view(-1, 1, nx).expand(-1, nt, nx)
            ).permute(0, 3, 1, 2)
        )
        pe = (
            torch.linspace(
                start=0,
                end=1,
                steps=nt,
                device=condition.device,
                dtype=condition.dtype.to_real(),
            )
            .view(1, 1, nt, 1)
            .expand(condition.n_samples, 1, nt, nx)
        )
        inputs.append(pe)
        return torch.cat(inputs, dim=1)

    def postprocess(
        self,
        u_pred: torch.Tensor,
        n_modes: int | None = None,
    ) -> torch.Tensor:
        """
        Post-process the predicted solution by applying a low-pass filter.

        Args:
            u_pred: torch.Tensor representing the raw output from the neural network.
            n_modes: Number of low pass filter modes to retain.

        Returns:
            A dict holding the complex solution field under the key ``u``, built
            from the two real output channels.

        """
        _, nx = self.resolution
        _, xmax = self.domain

        if n_modes is None:
            n_modes = nx // 2

        # setup spatial frequencies
        k = fft.rfftfreq(nx, d=xmax / (2 * torch.pi * nx)).to(u_pred)
        k_proj = torch.where(k.abs() <= torch.pi * n_modes / xmax, 1.0, 0.0).to(u_pred)

        # Fourier transform the solution
        v = fft.rfft(u_pred, norm="forward")

        # apply low-pass filter in Fourier space
        v_filtered = v * k_proj

        # transform back to regular space
        u_filtered = fft.irfft(v_filtered, norm="forward")

        # convert back to complex
        u_filtered = torch.view_as_complex(u_filtered.permute(0, 2, 3, 1).contiguous())

        return {"u": u_filtered}

    def solution_loss(
        self,
        solution_pred: EqSolution,
        solution_gt: EqSolution,
        normalizers: dict[str, torch.Tensor],
        stage: str = "",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the solution loss (manufactured-data loss, weight alpha_u).

        Args:
            solution_pred: the predicted solution field
            solution_gt: the regression target. In training it is the manufactured or
                synthesized solution u from the replay buffer, not a solver label;
                in evaluation it is the solver solution.
            normalizers: RMS normalizers for the solution and input fields
            stage: current stage of training, used in logging
        Returns:
            A tuple of the loss value and a dictionary of loss components

        """
        assert self.loss_weights is not None, "Loss weights not set."

        # extract normalizers
        u_norm = normalizers["u"]

        # compute the solution loss
        sol_loss = l2_norm(solution_pred["u"] - solution_gt["u"])
        sol_ic_loss = l2_norm(
            solution_pred["u"].narrow(1, 0, 1) - solution_gt["u"].narrow(1, 0, 1)
        )

        # compute the total loss
        loss = (
            self.loss_weights.get("sol", 0.0) * sol_loss / u_norm
            + self.loss_weights.get("sol_ic", 0.0) * sol_ic_loss / u_norm
        )

        return loss, {
            stage + "/sol_loss": sol_loss.item() / solution_pred.n_samples,
            stage + "/sol_ic_loss": sol_ic_loss.item() / solution_pred.n_samples,
        }

    def condition_loss(
        self,
        condition_pred: EqCondition,
        condition_gt: EqCondition,
        normalizers: dict[str, torch.Tensor],
        stage: str = "",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the PDE residual loss (weight alpha_v) between the realized input
        N(u_pred) and the sampled input v.

        Args:
            condition_pred: the realized input N(u_pred) of the predicted solution
            condition_gt: the sampled input v that the residual regresses onto
            normalizers: RMS normalizers for the input fields
            stage: current stage of training, used in logging
        Returns:
            A tuple of the loss value and a dictionary of loss components

        """
        assert self.loss_weights is not None, "Loss weights not set."

        # extract normalizers
        ic_norm = normalizers["ic"]
        f_norm = normalizers["f"]

        # compute the PDE residual (condition) loss
        cond_loss = l2_norm(condition_pred["f"] - condition_gt["f"])
        cond_h1_loss = h1_norm(condition_pred["f"] - condition_gt["f"])
        cond_ic_loss = l2_norm(condition_pred["ic"] - condition_gt["ic"])

        # compute the total loss
        loss = (
            self.loss_weights.get("cond", 0.0) * cond_loss / f_norm
            + self.loss_weights.get("cond_h1", 0.0) * cond_h1_loss / f_norm
            + self.loss_weights.get("cond_ic", 0.0) * cond_ic_loss / ic_norm
        )

        return loss, {
            stage + "/cond_loss": cond_loss.item() / condition_pred.n_samples,
            stage + "/cond_h1_loss": cond_h1_loss.item() / condition_pred.n_samples,
            stage + "/cond_ic_loss": cond_ic_loss.item() / condition_pred.n_samples,
        }

    def visualize(
        self,
        solution: EqSolution,
        condition: EqCondition,
        normalizers: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Visualize the solution and input data, assuming data is normalized to unit
        variance. Plot to ±3 standard deviations.

        Args:
            solution: EqSolution object representing the solution field
            condition: EqCondition holding the input v
            normalizers: dictionary of normalizers for the solution and input
        Returns:
            A tuple of numpy arrays representing the visualizations of the solution
            and of the forcing component of the input.

        """
        # unpack normalizers
        u_norm = normalizers["u"]
        f_norm = normalizers["f"]

        # normalize u and f
        u = solution["u"] / (1.5 * u_norm)
        f = condition["f"] / (1.5 * f_norm)

        # get u and f amplitude and phases
        u_amp = u.abs().clamp(0, 1)
        u_phase = (torch.angle(u) / (2 * torch.pi)) % 1
        f_amp = f.abs().clamp(0, 1)
        f_phase = (torch.angle(f) / (2 * torch.pi)) % 1

        # create hsv images
        gamma = 0.7
        u_hsv = torch.zeros((*u.shape, 3), device=u.device)
        u_hsv[..., 0] = u_phase
        u_hsv[..., 1] = 1.0
        u_hsv[..., 2] = u_amp**gamma
        f_hsv = torch.zeros((*f.shape, 3), device=f.device)
        f_hsv[..., 0] = f_phase
        f_hsv[..., 1] = 1.0
        f_hsv[..., 2] = f_amp**gamma

        # convert to rgb
        u_rgb = colors.hsv_to_rgb(u_hsv.cpu().numpy())
        f_rgb = colors.hsv_to_rgb(f_hsv.cpu().numpy())

        return u_rgb, f_rgb

    @classmethod
    def create_dataset(
        cls,
        alpha: float = 0.0,
        beta: float = -4.0,
        batch_size: int = 10,
        n_samples: int = 1000,
        domain: tuple[float, float] = (6.4, 50.0),
        resolution: tuple[int, int] = [128, 256],
        solver_domain: tuple[float, float] = (250.0, 50.0),
        solver_resolution: tuple[int, int] = [10000, 256],
        device: torch.device = torch.device("cpu"),
        seed: int | None = 0,
        burn_in: int = 1000,
        root_dir: str = "./data/gl_datasets",
        verbose: bool = False,
    ) -> None:
        """
        Create a dataset for the equation by solving from random initial
        conditions with no forcing term. Each trajectory is run past a burn-in so
        that the retained windows are samples of the invariant measure, then cut
        into consecutive windows of the target resolution.

        Args:
            alpha: The dispersion coefficient in (1 + i*alpha)Lap(u)
            beta: The nonlinear coefficient in (1 + i*beta)|u|^2u
            batch_size: Number of samples per batch
            n_samples: Total number of samples to generate
            domain: Desired domain size of the dataset
            resolution: Desired resolution of the dataset
            solver_domain: Domain size for the solver
            solver_resolution: Resolution for the solver
            device: Device to store the dataset
            seed: Random seed for reproducibility
            burn_in: Number of initial time steps to discard
            root_dir: Path to save the generated dataset
            verbose: Whether to display progress information

        """
        ratio_t = (domain[0] / resolution[0]) / (
            solver_domain[0] / solver_resolution[0]
        )
        ratio_x = (domain[1] / resolution[1]) / (
            solver_domain[1] / solver_resolution[1]
        )
        save_path = os.path.join(
            root_dir,
            f"alpha_{alpha:.0e}_beta_{beta:.0e}_"
            f"{domain[0]}t@{resolution[0]}_{domain[1]}x@{resolution[1]}.pt",
        )
        assert abs(ratio_t - round(ratio_t)) < 1e-8, (
            "the ratio of delta t between dataset and solver must be an integer."
        )
        assert abs(ratio_x - round(ratio_x)) < 1e-8, (
            "the ratio of delta x between dataset and solver must be an integer."
        )
        assert (solver_resolution[0] - burn_in) / ratio_t > resolution[0], (
            "the solver resolution is too low for the desired dataset resolution."
        )
        assert solver_domain[1] == domain[1], (
            "the spatial domain size for solver and dataset must be the same."
        )
        assert not os.path.exists(save_path), (
            "Dataset file already exists. Please remove it before creating a new one."
        )
        assert n_samples % batch_size == 0, "n_samples must be divisible by batch_size."
        ratio_t = round(ratio_t)
        ratio_x = round(ratio_x)

        # set random seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # calculate number of batches needed
        n_sample_per_traj = (solver_resolution[0] - burn_in) // (
            ratio_t * resolution[0]
        )
        n_batches = (n_samples - 1) // (n_sample_per_traj * batch_size) + 1
        pbar = trange(n_batches, desc="Generating dataset", disable=not verbose)

        # initialize equation instance, make the root directory if not exists
        os.makedirs(root_dir, exist_ok=True)
        equation = cls(
            root_dir=root_dir,
            alpha=alpha,
            beta=beta,
            domain=solver_domain,
            resolution=solver_resolution,
            fitting_samples=0,
            test_samples=0,
            loss_weights={},
            ot_rank=0,
        )

        samples = []
        for _ in pbar:
            # generate random initial condition
            ic = torch.randn(
                (batch_size, solver_resolution[1], 2),
                device=device,
                dtype=torch.float64,
            )
            ic = torch.view_as_complex(ic)
            f = torch.zeros(
                (batch_size, *solver_resolution),
                device=device,
                dtype=torch.complex128,
            )
            cond = EqCondition(
                {
                    "f": f,
                    "ic": ic,
                },
                domain=solver_domain,
            ).to(dtype=torch.complex128)

            # solve the equation with zero forcing term and no low-pass filter
            eq_solution = equation.backward(cond, M=64)

            # chunk the solution into smaller samples and save
            for i in range(n_sample_per_traj):
                start_idx = burn_in + i * ratio_t * resolution[0]
                end_idx = start_idx + ratio_t * resolution[0]
                sample_u = eq_solution["u"][:, start_idx:end_idx:ratio_t, ::ratio_x]

                # downsample to desired resolution, move to cpu to save memory
                samples.append(
                    EqSolution(
                        {"u": sample_u},
                        domain=domain,
                    ).to(device=torch.device("cpu"))
                )
                if len(samples) * batch_size >= n_samples:
                    break

        # save the dataset
        EqSolution.save(samples, save_path)
        pbar.close()
        if verbose:
            print(f"Dataset saved to {save_path}")
