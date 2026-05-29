import inspect
from typing import Any, Callable

import torch


def filter_kwargs(
    obj: object, config: dict[str, Any], exclude: list[str] = []
) -> dict[str, Any]:
    """
    Filter a config dictionary to only include arguments that are present in the
    signature of the given object.

    Args:
        obj: The object to filter arguments for.
        config: The configuration dictionary.
        exclude: A list of arguments to exclude from the filtered dictionary.

    Returns:
        A filtered configuration dictionary.

    """
    sig = inspect.signature(obj)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if (param.kind == param.POSITIONAL_OR_KEYWORD and param.name not in exclude)
    ]
    return {arg: config[arg] for arg in config if arg in filter_keys}


def update_config(config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Update a configuration dictionary with values from another configuration dictionary.

    Args:
        config: The configuration dictionary to update.
        override: The configuration dictionary to override with.

    Returns:
        The updated configuration dictionary.

    """
    for key, value in override.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                update_config(config[key], value)
            else:
                config[key] = value
    return config


class Registry:
    """
    A simple registry class for registering and retrieving objects.
    """

    def __init__(self, name, fallback_module=None):
        """
        Initialize the registry.

        Args:
            name: Display name used in error messages.
            fallback_module: Module to search when a key is not in the local dict.

        """
        self._name = name
        self._module_dict = {}
        self._fallback_module = fallback_module

    def register(self, name=None) -> Callable:
        """
        Register an object with the registry.

        Args:
            name: The name of the object to register.

        Returns:
            The object.

        """

        def _register(cls):
            registry_name = name if name is not None else cls.__name__
            if registry_name in self._module_dict:
                raise KeyError(f"{registry_name} is already registered in {self._name}")
            self._module_dict[registry_name] = cls
            return cls

        return _register

    def get(self, name):
        """
        Get an object from the registry.

        Args:
            name: The name of the object to get.

        Returns:
            The registered class or the attribute found on the fallback module.

        Raises:
            KeyError: If the name is not found in the registry or fallback module.

        """
        if name in self._module_dict:
            return self._module_dict[name]

        if hasattr(self._fallback_module, name):
            return getattr(self._fallback_module, name)

        raise KeyError(
            f"'{name}' not found in {self._name} or torch.optim. "
            f"Available custom: {list(self._module_dict.keys())}"
        )


optimizer_registry = Registry("OptimizerRegistry", torch.optim)
scheduler_registry = Registry("SchedulerRegistry", torch.optim.lr_scheduler)


@scheduler_registry.register("CosineWithWarmupLR")
def cosine_with_warmup_lr(
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    eta_min: float,
    warmup_steps: int,
    warmup_factor: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Cosine annealing learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer whose learning rate is scheduled.
        n_steps: Total number of training steps.
        eta_min: Minimum learning rate at the end of cosine decay.
        warmup_steps: Number of steps for the linear warmup phase.
        warmup_factor: Starting LR multiplier at step 0 of warmup.

    Returns:
        A ``SequentialLR`` that applies linear warmup followed by cosine annealing.

    """
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, end_factor=1.0, total_iters=warmup_steps
    )

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps - warmup_steps, eta_min=eta_min
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
    )
