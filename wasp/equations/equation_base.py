"""
Defines the base class for the PDEs used to manufacture training data.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import torch

from .equation_data import EqCondition, EqData, EqSolution


class EquationBase(ABC):
    """
    Base class for the nonlinear operator equations WaSP trains on. Specific
    equations inherit from this class and implement the abstract methods. PDEs are
    written as N(u) = v, where N is the PDE operator, u is the solution, and v is
    the input (initial state and forcing), stored as an EqCondition. Two methods
    carry the convention:
        - forward: given the solution u, derive the input v = N(u).
        - backward: given the input v, solve for the solution u = G(v).
    """

    solution_shapes: dict[str, tuple[int, ...]]
    condition_shapes: dict[str, tuple[int, ...]]
    solution_normalizer_key: list[str]
    condition_normalizer_key: list[str]
    domain: tuple[float, ...]
    resolution: tuple[int, ...]
    test_data: dict[str, EqData]

    def __init__(
        self,
        domain: tuple[float, ...],
        resolution: tuple[int, ...],
    ) -> None:
        """
        Initialize the EquationBase class.
        """
        super().__init__()
        self.domain = tuple(domain)
        self.resolution = tuple(resolution)
        self.test_data = {}

    def double(self) -> Self:
        """
        Convert the equation to double precision.
        """
        if self.dtype == torch.float32:
            self.dtype = torch.float64
        elif self.dtype == torch.complex64:
            self.dtype = torch.complex128
        elif self.dtype not in [torch.float64, torch.complex128]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return self

    def check_solution(self, solution: EqSolution) -> None:
        """
        Check if the solution is valid.
        """
        assert tuple(solution.domain) == self.domain, (
            "Solution domain does not match equation domain."
        )
        for key in self.solution_shapes:
            assert solution.shapes[key] == self.solution_shapes[key], (
                f"Solution key {key} has wrong shape: "
                f"{solution.shapes[key]} != {self.solution_shapes[key]}"
            )

    def check_condition(self, condition: EqCondition) -> None:
        """
        Check that the input v has the expected domain and field shapes.
        """
        assert tuple(condition.domain) == self.domain, (
            "Condition domain does not match equation domain."
        )
        for key in self.condition_shapes:
            assert condition.shapes[key] == self.condition_shapes[key], (
                f"Condition key {key} has wrong shape: "
                f"{condition.shapes[key]} != {self.condition_shapes[key]}"
            )

    @abstractmethod
    def setup(self, device: torch.device) -> None:
        """
        Setup the equation by loading the data to memory, computing statistics, and
        fitting necessary components for sampling solutions and mapping inputs.
        """
        pass

    @abstractmethod
    def forward(
        self,
        solution: EqSolution,
    ) -> EqCondition:
        """
        Apply the PDE operator to a solution, v = N(u).

        Args:
            solution: EqSolution object representing the solution field
        Returns:
            EqCondition holding the derived input v = (u(0, x), f(t, x))

        """
        pass

    @abstractmethod
    def backward(
        self,
        condition: EqCondition,
        *args,
        **kwargs,
    ) -> EqSolution:
        """
        Solve the equation for u = G(v) given the input v.

        Args:
            condition: EqCondition holding the input v
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        Returns:
            EqSolution object representing the solved solution field

        """
        pass

    @abstractmethod
    def sample_solution(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> EqSolution:
        """
        Draw solutions from the manufactured distribution mu_0.

        Args:
            n_samples: Number of samples to generate
            device: Device to store the sampled solutions
        Returns:
            EqSolution representing the sampled solutions

        """
        pass

    @abstractmethod
    def map_condition(self, condition: EqCondition, s: float) -> EqCondition:
        """
        Move the inputs along the McCann interpolation from the manufactured input
        distribution nu_0 at s = 0 to the target input distribution nu_1 at s = 1.

        Args:
            condition: EqCondition holding the inputs v
            s: interpolation parameter of the transport path, in [0, 1]

        Returns:
            EqCondition holding the interpolated inputs, distributed as nu_s

        """
        pass

    @abstractmethod
    def map_to_test_condition(self, solution: EqSolution) -> EqCondition:
        """
        Build the target input of a solution: its initial condition, zero forcing.

        Args:
            solution: EqSolution of solver trajectories
        Returns:
            EqCondition holding the target inputs v = (u(0, x), 0)

        """
        pass

    @abstractmethod
    def prepare_input(self, condition: EqCondition) -> torch.Tensor:
        """
        Prepare the input for the neural network.

        Args:
            condition: EqCondition holding the inputs v
        Returns:
            torch.Tensor representing the prepared input

        """
        pass

    @abstractmethod
    def postprocess(
        self, u_pred: torch.Tensor, *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Postprocess the output solution after solving the equation.

        Args:
            u_pred: torch.Tensor representing the raw output solution.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of the postprocessed solution.

        """
        pass

    @abstractmethod
    def solution_loss(
        self,
        solution_pred: EqSolution,
        solution_gt: EqSolution,
        normalizers: dict[str, float],
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
        pass

    @abstractmethod
    def condition_loss(
        self,
        condition_pred: EqCondition,
        condition_gt: EqCondition,
        normalizers: dict[str, float],
        stage: str = "",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the PDE residual loss (weight alpha_v) between the realized input
        N(u_pred) and the sampled input v.

        Args:
            condition_pred: the realized input N(u_pred) of the predicted solution
            condition_gt: the sampled input v that the residual regresses onto
            normalizers: RMS normalizers for the solution and input fields
            stage: current stage of training, used in logging
        Returns:
            A tuple of the loss value and a dictionary of loss components

        """
        pass

    @abstractmethod
    def create_dataset(
        cls,
        *args,
        **kwargs,
    ) -> None:
        """
        Create a dataset for the equation by generating and saving inputs and
        solutions to disk.
        """
        pass

    @abstractmethod
    def visualize(
        self,
        solution: EqSolution,
        condition: EqCondition,
        normalizers: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Visualize the solution and input data.

        Args:
            solution: EqSolution object representing the solution field
            condition: EqCondition holding the input v
            normalizers: RMS normalizers for the solution and input fields
        Returns:
            A tuple of numpy arrays representing the visualizations of the solution
            and of the forcing component of the input.

        """
        pass

    def _solve_coeff(
        self,
        device: torch.device,
        is_complex: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute and cache the ETDRK4 coefficients and differential operators.
        On first call, computes E, E_2, q, f1, f2, f3 from the cached linear
        operator L and time step dt. On subsequent calls, returns the cached
        values cast to the requested device.

        Subclasses must set self._etdrk4_L (the linear operator in Fourier
        space), self._etdrk4_dt (time step), and self._etdrk4_M (number of
        contour integration points) before calling this method.

        Args:
            device: Device to cast the coefficients to.
            is_complex: Whether the equation is complex-valued.

        Returns:
            Tuple of (E, E_2, q, f1, f2, f3) tensors.

        """
        if not hasattr(self, "_etdrk4_cache"):
            L = self._etdrk4_L
            dt = self._etdrk4_dt
            M = self._etdrk4_M

            # Precompute exponential operators
            E_2 = torch.exp(L * dt / 2)
            E = E_2.pow(2)

            # Setup contour integration points
            phase_scale = 2j if is_complex else 1j
            r = torch.exp(
                phase_scale
                * torch.pi
                * (
                    torch.arange(1, M + 1, device=L.device, dtype=L.dtype.to_real())
                    - 0.5
                )
                / M
            )

            # Evaluate contour integrals
            lr = dt * L.unsqueeze(-1) + r
            e_lr = torch.exp(lr / 2)
            e_lr2 = e_lr**2
            q = torch.mean(dim=-1, input=(e_lr - 1) / lr)
            f1 = torch.mean(
                dim=-1,
                input=(-4 - lr + e_lr2 * (4 - 3 * lr + lr**2)) / (lr**3),
            )
            f2 = torch.mean(dim=-1, input=(2 + lr + e_lr2 * (lr - 2)) / (lr**3))
            f3 = torch.mean(
                dim=-1,
                input=(-4 - 3 * lr - lr**2 + e_lr2 * (4 - lr)) / (lr**3),
            )

            # For real-valued equations, take the real part of the contour integrals
            if not is_complex:
                q, f1, f2, f3 = (
                    torch.real(q),
                    torch.real(f1),
                    torch.real(f2),
                    torch.real(f3),
                )

            # Scale coefficients by time step
            q, f1, f2, f3 = dt * q, dt * f1, dt * f2, dt * f3

            self._etdrk4_cache = (E, E_2, q, f1, f2, f3)

        return tuple(t.to(device) for t in self._etdrk4_cache)
