"""
Modified from https://github.com/fwilliams/scalable-pytorch-sinkhorn
such that the conventions align with https://arxiv.org/abs/2109.12004
to compute barycentric projections. Concretely, we define the distance
matrix to be 1/p * ||x - y||^p instead of ||x - y||_p.
"""

import warnings

import pykeops.torch as keops
import torch
import tqdm


class OTMapping:
    """
    Sinkhorn OT map with optional PCA acceleration and barycentric projection.

    Fits an entropy-regularized OT plan between two point clouds via the Sinkhorn
    algorithm (log-domain implementation using KeOps for memory efficiency), then
    applies barycentric projection to map new source points to the target space.

    Attributes:
        eps: Entropy regularization strength.
        max_iters: Maximum number of Sinkhorn iterations.
        stop_thresh: Convergence threshold on the max dual-variable change.
        q: PCA projection dimension, or None to skip PCA.
        verbose: Whether to print per-iteration progress.
        device: Device used for all computations.
        dtype: Floating-point dtype used for all computations.

    """

    v_j: keops.LazyTensor = None
    y_j: keops.LazyTensor = None
    x_rms: float = None
    y_rms: float = None

    def __init__(
        self,
        eps: float = 1e-3,
        max_iters: int = 100,
        stop_thresh: float = 1e-5,
        q: int | None = None,
        verbose: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the OTMapping class.

        Args:
            eps: entropy regularization term.
            max_iters: maximum number of Sinkhorn iterations to perform.
            stop_thresh: stopping threshold for Sinkhorn iterations.
            q: optional, for PCA-based acceleration, the dimension to project the
               point clouds onto before computing the OT mapping.
            verbose: if True, print out information on each Sinkhorn iteration.
            device: device to run the computations on.
            dtype: data type to use for computations.

        """
        self.device = device
        self.dtype = dtype

        if eps <= 0:
            raise ValueError("Entropy regularization term eps must be > 0")
        if max_iters <= 0:
            raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")
        if not isinstance(stop_thresh, float):
            raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

        self.eps = eps
        self.max_iters = max_iters
        self.stop_thresh = stop_thresh
        self.q = q
        self.verbose = verbose

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_se: bool = False,
    ) -> torch.Tensor:
        """
        Fit the OT mapping to the input point clouds x and y.

        Args:
            x: [n, d] tensor representing the first point cloud.
            y: [m, d] tensor representing the second point cloud.
            return_se: if True, return the standard error of the distance.

        Returns:
            The approximated 2-Wasserstein distance between the two point clouds.

        """
        x = x.to(self.device, self.dtype)
        y = y.to(self.device, self.dtype)

        if len(x.shape) != 2:
            raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
        if len(y.shape) != 2:
            raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}"
            )

        # Optional PCA-based acceleration
        if self.q is not None:
            if self.q <= 0 or self.q > x.shape[1]:
                raise ValueError(
                    f"q must be an integer in [1, {x.shape[1]}], got {self.q}"
                )

            y_mean = y.mean(dim=0, keepdim=True)
            y_centered = y - y_mean

            _, S, V = torch.pca_lowrank(y_centered, q=self.q, center=False)
            explained_variance = torch.sum(S**2) / (y.size(0) - 1)
            total_variance = torch.var(y_centered, dim=0, unbiased=True).sum()
            ratio = explained_variance / total_variance
            if ratio < 0.99:
                warnings.warn(
                    f"PCA projection onto {self.q} dimensions only explains "
                    f"{ratio:.4f} of the variance in y."
                )

            x = (x - y_mean) @ V  # [n, q]
            y = y_centered @ V  # [m, q]

            self.V = V
            self.y_mean = y_mean

        self.rms = y.square().sum(-1).mean().sqrt()
        x = x / self.rms
        y = y / self.rms

        x_i = keops.Vi(x)  # [n, 1, d]
        y_j = keops.Vj(y)  # [1, m, d]
        M_ij = ((x_i - y_j) ** 2).sum(dim=2) / 2  # [n, m]

        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]

        log_a = torch.log(w_x)  # [n]
        log_b = torch.log(w_y)  # [m]

        u = torch.zeros_like(w_x)
        v = self.eps * torch.log(w_y)
        u_i = keops.Vi(u.unsqueeze(-1))
        v_j = keops.Vj(v.unsqueeze(-1))

        pbar = tqdm.trange(self.max_iters, disable=not self.verbose)

        for _ in pbar:
            u_prev = u
            v_prev = v

            summand_u = (-M_ij + v_j) / self.eps
            u = self.eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
            u_i = keops.Vi(u.unsqueeze(-1))

            summand_v = (-M_ij + u_i) / self.eps
            v = self.eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
            v_j = keops.Vj(v.unsqueeze(-1))

            max_err_u = torch.max(torch.abs(u_prev - u))
            max_err_v = torch.max(torch.abs(v_prev - v))

            if self.verbose:
                pbar.set_postfix(
                    {"Current Max Error": max(max_err_u, max_err_v).item()}
                )
            if max_err_u < self.stop_thresh and max_err_v < self.stop_thresh:
                break

        # register for later use
        self.v_j = v_j
        self.y_j = y_j

        P_ij = ((-M_ij + u_i + v_j) / self.eps).exp()
        if x.shape[0] > y.shape[0]:
            cost = (P_ij * M_ij).sum(dim=1).sum()
        else:
            cost = (P_ij * M_ij).sum(dim=0).sum()

        S = self.rms / (x.shape[1] ** 0.5)
        distance = (2 * cost) ** 0.5
        final_distance = (distance * S).item()

        if return_se:
            var_u = torch.var(u, unbiased=True)
            var_v = torch.var(v, unbiased=True)
            n, m = x.shape[0], y.shape[0]
            se_cost = torch.sqrt(var_u / n + var_v / m)
            if final_distance > 0:
                uncertainty_se = ((S**2 / final_distance) * se_cost).item()
            else:
                uncertainty_se = 0.0
            return final_distance, uncertainty_se
        return final_distance

    def map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the barycentric projection of the optimal transport plan onto y
        Args:
            x: [n, d] tensor representing the new source point cloud.

        Returns:
            mapped_x: [n, d] tensor representing the mapped point cloud.

        """
        if self.v_j is None or self.y_j is None:
            raise ValueError("You must call fit() before calling map()")
        if len(x.shape) != 2:
            raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")

        if self.q is not None:
            x_pca = (x.to(self.device, self.dtype) - self.y_mean) @ self.V
            x_i = keops.Vi(x_pca / self.rms)
        else:
            x_i = keops.Vi(x.to(self.device, self.dtype) / self.rms)

        M_ij = ((x_i - self.y_j) ** 2).sum(dim=2) / 2  # [n, m]
        log_P_ij = (-M_ij + self.v_j) / self.eps  # [n, m]
        P_ij = (log_P_ij - keops.Vi(log_P_ij.max(dim=1))).exp()  # [n, m]
        yP_ij = self.y_j * P_ij  # [n, m, d]
        mapped_x = yP_ij.sum(dim=1) / P_ij.sum(dim=1)  # [n, d]

        if self.q is not None:
            return (self.rms * mapped_x @ self.V.T + self.y_mean).to(x)
        else:
            return (mapped_x * self.rms).to(x)
