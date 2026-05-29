"""
Define a circular replay buffer of solutions for the solution loss.

During warm-up (s = 0) it holds manufactured samples from mu_0; afterwards it holds
solutions synthesized by the operator on nu_s.
"""

import torch

from ..equations.equation_data import EqSolution


class ReplayBuffer:
    """
    Circular buffer that stores solution tensors for the solution loss, which
    constrains the operator to be a left inverse of the PDE operator.

    New entries overwrite the oldest when the buffer is full, keeping a sliding
    window of recent model predictions. Replaying stored predictions removes the
    serial dependency between the two loss terms, so both are evaluated in one
    forward pass, and keeps the regression targets fixed for several steps.

    Attributes:
        max_size: Maximum number of samples the buffer can hold.
        device: Device on which buffer tensors reside.
        dtype: Data type of stored tensors.
        domain: Physical domain extents associated with stored solutions.

    """

    def __init__(
        self,
        max_size: int,
        solution_shapes: dict[str, tuple[int, ...]],
        domain: tuple[float, ...] = (1.0, 1.0),
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the ReplayBuffer.

        Args:
            max_size: Maximum number of samples to store in the buffer.
            solution_shapes: Dictionary of solution shapes.
            domain: Physical domain size.
            device: Device to store the tensors on.
            dtype: Data type of the stored tensors.

        """
        self.max_size = max_size
        self.ptr = 0
        self._size = 0
        self.device = device
        self.dtype = dtype
        self.domain = tuple(domain)

        self.tensor_dict = {
            key: torch.zeros((max_size, *shape), device=device, dtype=dtype)
            for key, shape in solution_shapes.items()
        }

    @torch.no_grad()
    def insert(
        self,
        eq_solution: EqSolution,
        n: int | None = None,
    ) -> None:
        """
        Insert solutions into the buffer, overwriting the oldest entries.

        Args:
            eq_solution: The solutions to insert.
            n: Number of samples to insert; None inserts the whole batch.

        """
        if n is not None:
            eq_solution = eq_solution[:n]
        batch_size = eq_solution.n_samples

        if self.ptr + batch_size <= self.max_size:
            insert_slice = [
                (slice(self.ptr, self.ptr + batch_size), slice(0, batch_size))
            ]
        else:
            overflow = (self.ptr + batch_size) - self.max_size
            insert_slice = [
                (slice(self.ptr, self.max_size), slice(0, batch_size - overflow)),
                (slice(0, overflow), slice(batch_size - overflow, batch_size)),
            ]

        eq_solution = eq_solution.to(device=self.device, dtype=self.dtype)

        for buf_slice, data_slice in insert_slice:
            for key in self.tensor_dict.keys():
                self.tensor_dict[key][buf_slice] = eq_solution[key][data_slice].detach()

        self.ptr = (self.ptr + batch_size) % self.max_size
        self._size = min(self._size + batch_size, self.max_size)

    def sample(self, batch_size: int, last: bool = False) -> EqSolution:
        """
        Sample a batch of solutions from the buffer.

        Args:
            batch_size: Number of samples to draw.
            last: If True, sample the last `batch_size` inserted samples.

        Returns:
            An EqSolution object containing the sampled solutions.

        """
        if last:
            indices = torch.arange(
                (self.ptr - batch_size), self.ptr, device=self.device
            )
        else:
            indices = torch.randint(
                self.ptr - self._size, self.ptr, (batch_size,), device=self.device
            )

        sampled_tensor_dict = {
            key: self.tensor_dict[key][indices] for key in self.tensor_dict
        }

        eq_solution = EqSolution(
            sampled_tensor_dict,
            domain=self.domain,
        )

        return eq_solution

    def purge(self) -> None:
        """
        Clear the buffer.
        """
        self.ptr = 0
        self._size = 0

    def pop(self, num_samples: int) -> None:
        """
        Remove the oldest samples from the buffer.

        Args:
            num_samples: Number of samples to remove.

        """
        self._size -= num_samples

    @property
    def size(self):
        """
        Get the current size of the buffer.

        Returns:
            The number of samples currently stored in the buffer.

        """
        return self._size
