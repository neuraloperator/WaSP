import math
from typing import Literal


class TransportScheduler:
    """
    A scheduler for advancing the interpolation parameter s during training.

    s is the McCann interpolation parameter of nu_s. After a warm-up with s held at
    ``start_s``, s advances by ``increment`` once the normalized PDE residual loss
    falls below ``threshold``; a cosine or linear base schedule acts as a floor.
    """

    def __init__(
        self,
        enabled: bool,
        start_s: float,
        threshold: float,
        increment: float,
        min_sample: int,
        max_steps: int,
        adaptive_range: tuple[float, float],
        pre_training_ratio: float,
        base_schedule: Literal["linear", "cosine"] = "cosine",
    ) -> None:
        """
        Initialize the TransportScheduler.

        Args:
            enabled: Whether to use the transport scheduler.
            start_s: The starting value of s.
            threshold: Threshold on the mean normalized PDE residual loss below which
                s advances.
            increment: The step size by which the adaptive rule advances s.
            min_sample: The minimum number of samples before adjustment starts.
            max_steps: The maximum number of training steps.
            adaptive_range: The range of s in which the adaptive rule is active.
            pre_training_ratio: Fraction of total steps spent in warm-up, with s held
                at ``start_s``.
            base_schedule: The type of base schedule to use ("linear" or "cosine").

        """
        assert 0.0 < increment <= 1.0, "increment must be in (0, 1]"

        self.enabled = enabled
        self.threshold = threshold
        self.start_s = start_s
        self.increment = increment
        self.min_sample = min_sample
        self.end_pretraining = max_steps * pre_training_ratio
        self.training_steps = max_steps - self.end_pretraining
        self.base_schedule = base_schedule
        self.adaptive_range = adaptive_range
        self._s = start_s

        self.loss_sum = 0
        self.n_samples = 0

    def schedule_fn(self, step: int) -> float:
        """
        Compute s from the current step and the base schedule.

        Args:
            step: The current training step.

        Returns:
            The computed value of s in [0, 1].

        """
        # fraction of the post-warm-up budget spent so far
        r = (step - self.end_pretraining) / self.training_steps
        assert 0.0 <= r <= 1.0, f"r must be in [0, 1], but got {r}"
        if self.base_schedule == "linear":
            return self.start_s + (1 - self.start_s) * r
        elif self.base_schedule == "cosine":
            return self.start_s + (1 - self.start_s) * 0.5 * (1 - math.cos(math.pi * r))
        else:
            raise ValueError(f"Unknown base schedule: {self.base_schedule}")

    def step(self, loss: float, numel: int, step: int) -> None:
        """
        Update the scheduler with the latest loss values.

        Args:
            loss: The loss values from the current training step.
            numel: The number of elements in the current batch.
            step: The current training step.

        """
        if not self.enabled:
            return

        if step < self.end_pretraining:
            self._s = self.start_s
            return

        self.n_samples += numel
        self.loss_sum += loss

        if self.n_samples >= self.min_sample:
            mean_loss = self.loss_sum / self.n_samples
            if mean_loss < self.threshold and self.adaptive:
                self._s = min(self._s + self.increment, 1.0)
            elif self._s < self.schedule_fn(step):
                self._s = self.schedule_fn(step)
            self.loss_sum = 0
            self.n_samples = 0

    @property
    def adaptive(self) -> bool:
        """
        Check whether the adaptive rule is currently active.

        Returns:
            True if s lies strictly inside the adaptive range (open interval), so the
            rule is inactive at exactly s = 0 and at the upper bound.

        """
        return self.enabled and (
            self.adaptive_range[0] < self._s < self.adaptive_range[1]
        )

    @property
    def s(self) -> float:
        """
        Get the current interpolation parameter.

        Returns:
            The current value of s in [0, 1].

        """
        return self._s

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the scheduler.

        Returns:
            A dictionary containing the state of the scheduler.

        """
        return {
            "s": self._s,
            "loss_sum": self.loss_sum,
            "n_samples": self.n_samples,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary into the scheduler.

        Args:
            state_dict: A dictionary containing the state to load.

        """
        self._s = state_dict["s"]
        self.loss_sum = state_dict["loss_sum"]
        self.n_samples = state_dict["n_samples"]
