import os
import warnings

import numpy as np
import torch
from matplotlib import colormaps
from torch import fft
from tqdm import trange

from ..utils import FiniteDifference, h1_norm, l2_norm, sample_grf
from .equation_base import EquationBase
from .equation_data import EqCondition, EqData, EqSolution


class ACEquation(EquationBase):
    """
    The Allen-Cahn equation with an added forcing term:
        u_t - nu * u_xx + u^3 - u = f(t, x)
    where f is the forcing field. The input is v = (u(0, x), f(t, x)), stored
    as an EqCondition, which makes the PDE operator N a bijection.
    """

    test_data: dict[str, EqData] = {}
    solution_normalizer_key: list[str] = ["u"]
    condition_normalizer_key: list[str] = ["f", "ic"]

    def __init__(
        self,
        root_dir: str,
        nu: float,
        ic_amplitude: float,
        power_decay: float,
        domain: tuple[float, float],
        resolution: tuple[int, int],
        loss_weights: dict[str, float],
        test_samples: int,
    ) -> None:

        super().__init__(
            domain=domain,
            resolution=resolution,
        )

        self.solution_shapes = {"u": resolution}
        self.condition_shapes = {"f": resolution, "ic": (resolution[1],)}

        self.root_dir = root_dir
        self.nu = nu
        self.ic_amplitude = ic_amplitude
        self.power_decay = power_decay
        self.loss_weights = loss_weights
        self.test_samples = test_samples

        self.finite_diff = FiniteDifference(N=5, degree=4)

    def setup(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Load the test dataset and build the initial-condition power spectrum.
        The spectrum is analytic here, so no transport map is fitted: the target
        initial-condition distribution is the Gaussian random field the
        manufactured samples already start from.

        Args:
            device: Device to store the fitted statistics and mappings

        """
        assert os.path.exists(self.root_dir), (
            f"Root directory {self.root_dir} does not exist."
        )

        print("Setting up AC equation...")
        print("Loading test datasets...")

        # load the solver dataset and hold out the test split; the power spectrum
        # is analytic and no transport map is fitted
        filename = os.path.join(
            self.root_dir,
            f"nu_{self.nu:.0e}_amp_{self.ic_amplitude:.0e}_decay_{self.power_decay:.1e}"
            f"_{self.domain[0]}t@{self.resolution[0]}"
            f"_{self.domain[1]}x@{self.resolution[1]}.pt",
        )
        if os.path.exists(filename):
            eq_solution = EqSolution.load(filename)
            # split the dataset into test and fitting
            sample_indices = torch.randperm(eq_solution.n_samples)
            self.test_data = {
                "solution": eq_solution[sample_indices[: self.test_samples]]
            }
        else:
            warnings.warn(f"Test dataset file {filename} does not exist.")
            self.test_data = {}

        # extract parameters
        _, nx = self.resolution
        _, xmax = self.domain
        kx = torch.fft.rfftfreq(nx, d=xmax / (2 * torch.pi * nx))
        self.ic_power_spectrum = self.ic_amplitude * (
            torch.exp(-self.power_decay * kx) * self.power_decay * nx / xmax
        )
        self.ic_power_spectrum[0] = 0  # zero DC component

    def forward(self, solution: EqSolution) -> EqCondition:
        """
        Apply the PDE operator to a solution, v = N(u).

        Args:
            solution: EqSolution object representing the solution field
        Returns:
            EqCondition holding the derived input v = (u(0, x), f(t, x))

        """
        self.check_solution(solution)

        # cast to float64 for better accuracy in fft operations
        dtype = solution.dtype
        solution = solution.to(dtype=torch.float64)

        # extract parameters
        nt, nx = self.resolution
        tmax, xmax = self.domain
        u = solution["u"]

        # setup spatial frequencies and time derivative module
        k = fft.rfftfreq(nx, d=xmax / (2 * torch.pi * nx)).to(u)

        # Fourier transform the solution (v holds rfft(u) here, not the input v)
        v = fft.rfft(u, norm="forward")

        # compute linear and nonlinear terms in Fourier space
        # u_xx -> -k^2 * v, nonlinear term u^3
        linear_term = self.nu * k.pow(2) * v - v
        nonlinear_term = fft.rfft(u.pow(3), norm="forward")

        # compute time derivative in regular space
        time_derivative = self.finite_diff.derive(u, delta=tmax / nt, dim=1)

        # transform spatial derivative back to regular space
        spatial_derivative = fft.irfft(
            linear_term + nonlinear_term, n=nx, norm="forward"
        )

        # compute forcing term f(t, x)
        f = time_derivative + spatial_derivative

        # return the input v as an EqCondition
        return EqCondition(
            {"f": f.to(dtype), "ic": u[:, 0, :].to(dtype)}, domain=self.domain
        )

    def backward(
        self,
        condition: EqCondition,
        M: int | None = 64,
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

        # cast to float64
        dtype = condition.dtype
        condition = condition.to(dtype=torch.float64)

        # setup spatial frequencies
        k = fft.rfftfreq(nx, d=xmax / (2 * torch.pi * nx)).to(device)

        # Compute and cache ETDRK4 coefficients
        if not hasattr(self, "_etdrk4_cache"):
            self._etdrk4_L = -self.nu * k.pow(2) + 1
            self._etdrk4_dt = dt
            self._etdrk4_M = M
        E, E_2, q, f1, f2, f3 = self._solve_coeff(device)

        # define stepping functions
        def nonlinear_term_spectral(hat_u: torch.Tensor) -> torch.Tensor:
            u = fft.irfft(hat_u, n=nx, norm="forward")
            nonlinear_term = -u.pow(3)
            out = fft.rfft(nonlinear_term, norm="forward")
            return out

        def linear_term_exponential_time(hat_u: torch.Tensor) -> torch.Tensor:
            return E * hat_u

        # time-stepping loop
        u = condition["ic"]
        u_history = torch.empty((u.size(0), nt, nx), device=device, dtype=u.dtype)
        u_history[:, 0, :] = u

        # Transform initial condition and forcing term to Fourier space
        # upsample f to match ETDRK4 time steps
        f_fft = fft.rfft(condition["f"], norm="forward")
        v = fft.rfft(u, norm="forward")

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

            # Store solution at current time step
            u_history[:, n, :] = fft.irfft(v, n=nx, norm="forward")

        return EqSolution({"u": u_history.to(dtype=dtype)}, domain=self.domain)

    def sample_solution(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> EqSolution:
        """
        Draw solutions from the manufactured distribution mu_0: sample the target
        initial condition, then take one Lie-Trotter splitting step of size t for
        each output time t, solving the reaction ODE u_t = u - u^3 in closed form
        and applying the diffusion semigroup exactly in Fourier space.

        Args:
            n_samples: Number of samples to generate
            device: Device to store the sampled solutions
        Returns:
            EqSolution object representing the sampled solutions

        """
        assert self.ic_power_spectrum is not None, (
            "Initial condition power spectrum not computed. "
            "Please run setup() before sampling solutions."
        )
        tmax, xmax = self.domain
        nt, nx = self.resolution
        ts = torch.linspace(0, tmax, nt + 1, device=device)[1:].view(1, -1, 1)
        kx = torch.fft.rfftfreq(nx, d=xmax / (2 * torch.pi * nx), device=device)
        L = -self.nu * kx.pow(2)

        # sample from Gaussian random field with the calculated power spectrum
        u_0 = sample_grf(self.ic_power_spectrum, n_samples, (nx,), device=device)

        # solve for non-diffusive solution
        u0 = u_0.view(n_samples, 1, nx)
        u_non_diff = u0 / torch.sqrt(
            u0.square() + (1 - u0.square()) * torch.exp(-2 * ts)
        )

        # apply diffusion by exponential-time operator in Fourier space
        u_diff = torch.fft.rfft(u_non_diff, norm="forward") * torch.exp(L * ts)
        u_diff = torch.fft.irfft(u_diff, n=nx, norm="forward")

        # return the sampled solutions
        return EqSolution({"u": u_diff}, domain=self.domain)

    def map_condition(self, condition: EqCondition, s: float) -> EqCondition:
        """
        Move the inputs along the McCann interpolation. No transport map is fitted:
        the manufactured and target initial-condition distributions coincide, so
        the initial condition is unchanged and the forcing anneals as (1 - s) f.

        Args:
            condition: EqCondition holding the inputs v
            s: interpolation parameter of the transport path, in [0, 1]

        Returns:
            EqCondition holding the interpolated inputs, distributed as nu_s

        """
        self.check_condition(condition)
        assert 0.0 <= s <= 1.0, "s must be in the range [0, 1]."

        return condition.new_condition(
            {
                "f": (1 - s) * condition["f"],
                "ic": condition["ic"],
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

        inputs = [condition["f"].view(-1, 1, nt, nx)]
        inputs.append(condition["ic"].view(-1, 1, 1, nx).expand(-1, 1, nt, nx))
        pe = (
            torch.linspace(
                start=0,
                end=1,
                steps=nt,
                device=condition.device,
                dtype=condition.dtype,
            )
            .view(1, 1, nt, 1)
            .expand(condition.n_samples, 1, nt, nx)
        )
        inputs.append(pe)
        return torch.cat(inputs, dim=1)

    def postprocess(
        self,
        u_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Post-process the raw model output. No spectral filtering is applied for
        this equation; the single output channel is unpacked.

        Args:
            u_pred: torch.Tensor representing the raw output from the neural network.

        Returns:
            A dict holding the solution field under the key ``u``.

        """
        return {"u": u_pred[:, 0, :, :]}

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
        normalizers: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Visualize the solution and input data, assuming data is normalized to unit
        variance. Plot to ±3 standard deviations.

        Args:
            solution: EqSolution object representing the solution field
            condition: EqCondition holding the input v
            normalizers: tuple of normalizers for the solution and input
        Returns:
            A tuple of numpy arrays representing the visualizations of the solution
            and of the forcing component of the input.

        """
        # unpack normalizers
        f_norm = normalizers["f"]

        # normalize to [0, 1] for visualization
        u_vis = torch.clamp(solution["u"] / 2 + 0.5, 0.0, 1.0).cpu()
        f_vis = torch.clamp(condition["f"] / f_norm / 6 + 0.5, 0.0, 1.0).cpu()

        cmap = colormaps["jet"]

        return cmap(u_vis)[..., :3], cmap(f_vis)[..., :3]

    @classmethod
    def create_dataset(
        cls,
        nu: float = 0.0001,
        ic_amplitude: float = 1e-3,
        power_decay: float = 0.5,
        batch_size: int = 1,
        n_samples: int = 1000,
        domain: tuple[float, float] = (5.0, 2.0),
        resolution: tuple[int, int] = [128, 256],
        solver_resolution: tuple[int, int] = [256, 512],
        device: torch.device = torch.device("cpu"),
        seed: int | None = 0,
        root_dir: str = "./data/ac_datasets",
    ) -> None:
        """
        Create a dataset for the equation by solving from random initial conditions
        with no forcing term, then downsampling to the target resolution.

        Args:
            nu: The diffusion coefficient of the AC equation
            ic_amplitude: Amplitude of the initial-condition power spectrum
            power_decay: Decay rate of the initial-condition power spectrum,
                P(k) proportional to exp(-power_decay * k)
            batch_size: Number of samples per batch
            n_samples: Total number of samples to generate
            domain: Desired domain size of the dataset
            resolution: Desired resolution of the dataset
            solver_resolution: Resolution for the solver
            device: Device to store the dataset
            seed: Random seed for reproducibility
            root_dir: Path to save the generated dataset

        """
        ratio_t = solver_resolution[0] / resolution[0]
        ratio_x = solver_resolution[1] / resolution[1]
        save_path = os.path.join(
            root_dir,
            f"nu_{nu:.0e}_amp_{ic_amplitude:.0e}_decay_{power_decay:.1e}_"
            f"{domain[0]}t@{resolution[0]}_{domain[1]}x@{resolution[1]}.pt",
        )
        assert abs(ratio_t - round(ratio_t)) < 1e-8, (
            "the ratio of delta t between dataset and solver must be an integer."
        )
        assert abs(ratio_x - round(ratio_x)) < 1e-8, (
            "the ratio of delta x between dataset and solver must be an integer."
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
        n_batches = (n_samples - 1) // (batch_size) + 1
        pbar = trange(n_batches, desc="Generating dataset")

        # initialize equation instance, make the root directory if not exists
        os.makedirs(root_dir, exist_ok=True)
        equation = cls(
            root_dir=root_dir,
            nu=nu,
            ic_amplitude=ic_amplitude,
            power_decay=power_decay,
            domain=domain,
            resolution=solver_resolution,
            test_samples=0,
            loss_weights={},
        )
        equation.setup(device=device)

        samples = []
        for _ in pbar:
            # generate random initial condition
            sol = equation.sample_solution(batch_size)
            cond = equation.map_to_test_condition(sol)

            # solve the equation with zero forcing term and no low-pass filter
            eq_solution = equation.backward(cond, M=64)

            # downsample to desired resolution, move to cpu to save memory
            samples.append(
                EqSolution(
                    {"u": eq_solution["u"][:, ::ratio_t, ::ratio_x].cpu()},
                    domain=domain,
                ).to(device="cpu")
            )

        # save the dataset
        EqSolution.save(samples, save_path)
        pbar.close()
