import os
import warnings

import torch
import torch.distributed as dist
from torch import nn


class CheckpointManager:
    """
    Save and load training checkpoints, with optional DDP state-dict normalization.

    Handles directory creation, overwrites warnings, and transparently strips or
    adds the ``module.`` prefix that DistributedDataParallel introduces in state
    dicts so that checkpoints are portable between distributed and single-GPU runs.

    Attributes:
        checkpoint_dir: Root directory where checkpoint files are written.
        save_checkpoints: When False all save calls are no-ops, useful for debug runs.

    """

    def __init__(self, checkpoint_dir: str, save_checkpoints: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints

    def save(self, state_dict: dict, step: int, name: str | None = None) -> None:
        """
        Save a state dict to ``<checkpoint_dir>[/<name>]/model_step_<step>.pt``.

        Args:
            state_dict: Arbitrary dictionary to serialize (model weights, optimizer
                state, config, etc.).
            step: Training step number, embedded in the filename.
            name: Optional subdirectory name under ``checkpoint_dir``.

        """
        if not self.save_checkpoints:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if name:
            save_path = os.path.join(self.checkpoint_dir, name, f"model_step_{step}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = os.path.join(self.checkpoint_dir, f"model_step_{step}.pt")

        if os.path.exists(save_path):
            warnings.warn(f"Overwriting existing model at {save_path}")

        torch.save(state_dict, save_path)

    @staticmethod
    def load(path: str, map_location: torch.device = torch.device("cpu")) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            map_location: Device to map tensors onto when loading.

        Returns:
            The deserialized state dict.

        """
        return torch.load(path, map_location=map_location, weights_only=False)

    @staticmethod
    def get_model_state(model: nn.Module) -> dict:
        """
        Get the model state dictionary.

        Args:
            model: The model to get the state from.

        Returns:
            The state dictionary.

        """
        if dist.is_initialized():
            return model.module.state_dict()
        return model.state_dict()

    @staticmethod
    def load_model_state(model: nn.Module, state_dict: dict) -> None:
        """
        Load the model state from the specified path.

        Args:
            model: The model to load the state into.
            state_dict: The state dictionary to load.

        """
        if dist.is_initialized():
            # If model is DDP, expect "module." prefix in state_dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("module."):
                    new_state_dict["module." + k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            # If model is NOT distributed, but checkpoint IS distributed (has module.),
            # strip it.
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
