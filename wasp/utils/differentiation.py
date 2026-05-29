from typing import Literal

import torch
import torch.nn.functional as F
from sympy import Matrix, factorial


class FiniteDifference:
    """
    Finite difference stencil for numerical differentiation.
    """

    def __init__(self, N: int, degree: int = None) -> None:
        """
        Initialize the finite difference stencil.

        Args:
            N (int): Number of points in the stencil (must be odd).
            degree (int): Polynomial degree to fit.

        """
        assert N % 2 == 1, "N must be odd"

        if degree is None:
            degree = N - 1

        assert N >= degree - 1, "N must be at least degree - 1"

        self.N = N
        self.degree = degree

        # Solve for central, forward, and backward stencils
        # Shape: (N // 2, degree + 1, N) for forward/backward
        self.coeffs = {
            "forward": torch.empty(N // 2, degree + 1, N, dtype=torch.float64),
            "backward": torch.empty(N // 2, degree + 1, N, dtype=torch.float64),
        }
        for i in range(N // 2):
            self.coeffs["forward"][i] = self._solve_coefficients(range(-i, N - i))
        for i in range(-N // 2 + 1, 0):
            self.coeffs["backward"][i] = self._solve_coefficients(range(-N - i, -i))
        # Shape: (degree + 1, N) for central
        self.coeffs["central"] = self._solve_coefficients(
            range(-N // 2 + 1, N // 2 + 1)
        )

    def _solve_coefficients(
        self, points: list[float], dtype=torch.float64
    ) -> torch.Tensor:
        """
        Solve for stencil coefficients using a polynomial of degree `degree`.

        Args:
            points (list[float]): Points to use for the stencil.
            dtype (torch.dtype): Data type to use for the stencil.

        Returns:
            torch.Tensor: Stencil coefficients.

        """
        # Convert points to a SymPy column matrix
        X = Matrix(
            [[p**k / factorial(k) for k in range(self.degree + 1)] for p in points]
        )

        # Compute (X^T X)^-1 X^T
        XtX_inv = (X.T * X).inv()
        stencil_matrix = XtX_inv * X.T

        # The stencil for all derivative orders
        stencil = torch.tensor(stencil_matrix.tolist(), dtype=dtype)

        return stencil

    def derive(
        self,
        u: torch.Tensor,
        delta: float,
        order: int = 1,
        dim: int = 0,
        boundary: Literal["periodic", "even", "odd", "none"] = "none",
    ) -> torch.Tensor:
        """
        Apply the finite difference stencil to the input tensor.

        Args:
            u (torch.Tensor): Input tensor.
            delta (float): Grid spacing.
            order (int): Derivative order.
            dim (int): Dimension along which to compute the derivative.
            boundary (str): Boundary type for the convolution. Note that for even/odd
                boundaries, it is always assumed that the even/odd boundary is at the
                0-th index edge, and the edge point is sampled.

        Returns:
            torch.Tensor: Derivative of the input tensor.

        """
        assert u.dim() >= 1, "Input tensor must have at least one dimension"
        assert u.size(dim) >= self.N, (
            f"Input tensor size along dimension {dim} must be at least {self.N}"
        )
        assert order <= self.degree, (
            f"Order {order} must be less than degree {self.degree}"
        )
        assert boundary in ["periodic", "even", "odd", "none"]

        if dim != 0:
            u = u.transpose(0, dim)

        # make sure coefficients are on the same device and dtype as u
        central = self.coeffs["central"][order].to(u)
        forward = self.coeffs["forward"][:, order].to(u)
        backward = self.coeffs["backward"][:, order].to(u)

        conv_u = u.reshape(u.size(0), 1, -1).transpose(0, 2)
        du = torch.zeros_like(u)

        # If specified, use padding mode
        if boundary == "periodic":
            conv_u = F.pad(conv_u, pad=(self.N // 2, self.N // 2), mode="circular")
            conv_slice = slice(0, u.size(0))
        elif boundary == "even":
            conv_u = F.pad(conv_u, pad=(self.N // 2, 0), mode="reflect")
            conv_slice = slice(0, u.size(0) - self.N // 2)
        elif boundary == "odd":
            conv_u = F.pad(conv_u, pad=(self.N // 2, 0), mode="reflect")
            conv_u[..., : self.N // 2] *= -1  # flip the sign of the reflected values
            conv_slice = slice(0, u.size(0) - self.N // 2)
        elif boundary == "none":
            conv_slice = slice(self.N // 2, u.size(0) - self.N // 2)
        else:
            raise ValueError(f"Unknown boundary type: {boundary}")

        # build slicing for central stencil and shape
        conv_shape = (conv_slice.stop - conv_slice.start, *u.shape[1:])

        # Convolve with central stencil
        conv_u = torch.nn.functional.conv1d(
            conv_u, central.reshape(1, 1, -1), padding=0
        )
        du[conv_slice] = conv_u.transpose(0, 2).reshape(conv_shape)

        # If padding was none, override with forward differences
        if boundary == "none":
            du[: self.N // 2] = torch.einsum("ij,j...->i...", forward, u[: self.N])
        # If padding was none, override with backward differences
        if boundary in ["even", "odd", "none"]:
            du[-self.N // 2 + 1 :] = torch.einsum(
                "ij,j...->i...", backward, u[-self.N :]
            )

        if dim != 0:
            du = du.transpose(0, dim)

        return du.div(delta**order)
