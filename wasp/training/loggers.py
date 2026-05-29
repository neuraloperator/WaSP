import datetime
import warnings

import wandb
from neuralop.utils import get_wandb_api_key


class WandBLogger:
    """
    Batched W&B logger that accumulates scalars and images between commit calls.

    Scalars are averaged over the window between commits; images are concatenated.
    Logging is a no-op when ``enabled`` is False, so the same code path works in
    debug or no-wandb runs without branching in the caller.

    Attributes:
        enabled: Whether W&B logging is active.
        log_every: Number of training steps between flushes to W&B.

    """

    def __init__(
        self, config: dict, n_params: int, enabled: bool = False, log_every: int = 50
    ):
        self.enabled = enabled and config.wandb.log
        self.log_every = log_every

        self._scalar_cache: dict[str, list[float]] = {}
        self._image_cache: dict[str, list] = {}
        self._steps_since_commit: int = 0
        self._last_commit_step: int | None = None

        if self.enabled:
            wandb.login(key=get_wandb_api_key())
            if config.wandb.name:
                wandb_name = config.wandb.name
            else:
                wandb_name = datetime.datetime.now().strftime("%Y%m-%d%H-%M%S-run")

            wandb_init_args = dict(
                config=config,
                name=wandb_name,
                group=config.wandb.group,
                project=config.wandb.project,
                entity=config.wandb.entity,
            )
            self.run = wandb.init(**wandb_init_args)
            self.run.summary["n_params"] = n_params
        else:
            self.run = None

    def log(self, data: dict, step: int) -> None:
        """
        Accumulate scalar values into the internal cache.
        Values are averaged and flushed on commit().
        """
        if not self.enabled:
            return

        # verify consistent step parameter within a commit window
        if self._last_commit_step is not None and step != self._last_commit_step:
            raise ValueError(
                f"Inconsistent step parameter within commit window: "
                f"expected {self._last_commit_step}, got {step}. "
                f"All log() calls between commits must use the same step."
            )
        self._last_commit_step = step

        for k, v in data.items():
            if k not in self._scalar_cache:
                self._scalar_cache[k] = []
            self._scalar_cache[k].append(v)

    def log_images(
        self,
        key: str,
        images: list,
        step: int,
        start_idx: int = 0,
        caption_prefix: str = "Sample",
    ) -> None:
        """
        Create wandb.Image objects and cache them internally.
        Logs a warning if the same key is logged more than once per commit window.
        """
        if not self.enabled:
            return

        # verify consistent step parameter within a commit window
        if self._last_commit_step is not None and step != self._last_commit_step:
            raise ValueError(
                f"Inconsistent step parameter within commit window: "
                f"expected {self._last_commit_step}, got {step}. "
                f"All log() calls between commits must use the same step."
            )
        self._last_commit_step = step

        img_list = [
            wandb.Image(images[i], caption=f"{caption_prefix} {i + start_idx}")
            for i in range(len(images))
        ]

        if key in self._image_cache:
            warnings.warn(
                f"Image key '{key}' already logged in the current commit window. "
                f"Concatenating images.",
                stacklevel=2,
            )
            self._image_cache[key].extend(img_list)
        else:
            self._image_cache[key] = img_list

    def commit(self, step: int) -> None:
        """
        Signal that the trainer has finished logging all info for this step.
        Increments the internal counter; when log_every steps have been
        accumulated, averages scalar values, merges image lists, flushes
        everything to wandb, and resets the caches.
        """
        if not self.enabled:
            return

        self._steps_since_commit += 1
        self._last_commit_step = None

        if self._steps_since_commit >= self.log_every:
            combined: dict = {}
            for k, vals in self._scalar_cache.items():
                combined[k] = sum(vals) / len(vals)
            combined.update(self._image_cache)

            wandb.log(combined, step=step)

            self._scalar_cache.clear()
            self._image_cache.clear()
            self._steps_since_commit = 0

    def finish(self) -> None:
        """Flush any remaining cached data and mark the W&B run as finished."""
        if self.enabled:
            if self._scalar_cache or self._image_cache:
                combined: dict = {}
                for k, vals in self._scalar_cache.items():
                    combined[k] = sum(vals) / len(vals)
                combined.update(self._image_cache)
                wandb.log(combined)
                self._scalar_cache.clear()
                self._image_cache.clear()
            wandb.finish()
