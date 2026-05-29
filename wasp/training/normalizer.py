import torch
from torch import nn

from ..equations import EqCondition, EqSolution


class Normalizer(nn.Module):
    """
    Online EMA normalizer tracking per-field RMS as the training distribution shifts.

    Inputs to the model are divided by the running RMS before the forward pass;
    outputs are multiplied back before loss computation. Updates are gated by an
    ``enabled`` flag so early out-of-distribution samples do not corrupt the statistics.

    Attributes:
        solution_keys: Field names belonging to the solution.
        condition_keys: Field names belonging to the input v (the condition).
        momentum: EMA update rate per sample.
        initialized: Whether the buffers have been seeded with the first batch.

    """

    def __init__(
        self,
        solution_keys: list[str],
        condition_keys: list[str],
        momentum: float = 1e-3,
    ) -> None:
        super().__init__()

        assert all(key not in condition_keys for key in solution_keys), (
            "Solution and condition keys must be disjoint."
        )

        self.solution_keys = solution_keys
        self.condition_keys = condition_keys
        self.momentum = momentum
        self.initialized = False

        self.keys = self.solution_keys + self.condition_keys

        for key in self.keys:
            self.register_buffer(key, torch.ones(1), persistent=True)

    @torch.no_grad()
    def update(
        self,
        solution: EqSolution,
        condition: EqCondition,
        enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Update the running statistics using the new batch data.

        Args:
            solution: EqSolution holding a batch of solution fields.
            condition: EqCondition holding a batch of input fields.
            enabled: whether to update the running statistics

        Returns:
            A dict of the current per-field RMS values, for logging.

        """
        for key in self.solution_keys:
            mean_sq = solution[key].abs().square().mean()
            if not self.initialized:
                getattr(self, key).copy_(mean_sq)
            elif enabled:
                kappa = (1 - self.momentum) ** solution.n_samples
                getattr(self, key).mul_(kappa).add_(mean_sq * (1 - kappa))

        for key in self.condition_keys:
            mean_sq = condition[key].abs().square().mean()
            if not self.initialized:
                getattr(self, key).copy_(mean_sq)
            elif enabled:
                kappa = (1 - self.momentum) ** condition.n_samples
                getattr(self, key).mul_(kappa).add_(mean_sq * (1 - kappa))

        if not self.initialized:
            self.initialized = True

        return {
            f"normalizers/{key}_rms": getattr(self, key).sqrt().item()
            for key in self.keys
        }

    def normalize_sol(self, solution: EqSolution) -> EqSolution:
        """
        Normalize the solution fields using the running statistics.

        Args:
            solution: EqSolution holding a batch of solution fields.

        Returns:
            A new EqSolution with each tracked field divided by its RMS.

        """
        tensor_dict = {}
        for key in self.solution_keys:
            tensor_dict[key] = solution[key] / getattr(self, key).sqrt()
        return solution.new_solution(tensor_dict)

    def normalize_cond(self, condition: EqCondition) -> EqCondition:
        """
        Normalize the input fields v using the running statistics.

        Args:
            condition: EqCondition holding a batch of input fields.

        Returns:
            A new EqCondition with each tracked field divided by its RMS.

        """
        tensor_dict = {}
        for key in self.condition_keys:
            tensor_dict[key] = condition[key] / getattr(self, key).sqrt()
        return condition.new_condition(tensor_dict)

    def denormalize_sol(self, solution: EqSolution) -> EqSolution:
        """
        Denormalize the solution fields using the running statistics.

        Args:
            solution: EqSolution holding a batch of solution fields.

        Returns:
            A new EqSolution with each tracked field multiplied by its RMS.

        """
        tensor_dict = {}
        for key in self.solution_keys:
            tensor_dict[key] = solution[key] * getattr(self, key).sqrt()
        return solution.new_solution(tensor_dict)

    def denormalize_cond(self, condition: EqCondition) -> EqCondition:
        """
        Denormalize the input fields v using the running statistics.

        Args:
            condition: EqCondition holding a batch of input fields.

        Returns:
            A new EqCondition with each tracked field multiplied by its RMS.

        """
        tensor_dict = {}
        for key in self.condition_keys:
            tensor_dict[key] = condition[key] * getattr(self, key).sqrt()
        return condition.new_condition(tensor_dict)

    @property
    def stats(self) -> dict[str, float]:
        """
        Get the current statistics of the normalizer.

        Returns:
            A dictionary containing the statistics of the normalizer.

        """
        return {key: getattr(self, key).sqrt().item() for key in self.keys}
