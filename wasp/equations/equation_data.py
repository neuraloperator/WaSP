"""
Defines the containers for PDE solution and input fields.

The input v = (u(0, x), f(t, x)) is stored as an EqCondition.
"""

from __future__ import annotations

from typing import Self, overload

import torch


class EqData:
    """
    Data structure to hold the input or the solution of an equation.
    Supports moving to different devices and dtypes.
    """

    shapes: dict[str, tuple[int, ...]]
    tensor_dict: dict[str, torch.Tensor]
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        tensor_dict: dict[str, torch.Tensor],
        domain: tuple[float, ...],
    ) -> None:
        """
        Initialize the EqData object.

        Args:
            tensor_dict: Dictionary containing the input tensors.
            domain: Domain of the equation.

        """
        device = list(tensor_dict.values())[0].device
        dtype = list(tensor_dict.values())[0].dtype
        n_samples = list(tensor_dict.values())[0].size(0)
        shapes = {}

        for key, tensor in tensor_dict.items():
            assert isinstance(tensor, torch.Tensor), (
                "Input tensors must be torch.Tensor objects."
            )
            assert tensor.dtype == dtype, "Input tensors must have the same dtype."
            assert tensor.device == device, "Input tensors must have the same device."
            assert tensor.size(0) == n_samples, (
                "Input tensors must have the same batch size."
            )
            shapes[key] = tuple(tensor.shape[1:])  # remove batch dimension

        self.tensor_dict = tensor_dict
        self.shapes = shapes
        self.device = device
        self.dtype = dtype
        self.domain = domain
        self.n_samples = n_samples

    def __str__(self) -> str:
        dict_str = ", ".join(
            [f"{key}: {self.tensor_dict[key].shape}" for key in self.keys]
        )
        return f"{self.__class__.__name__}({{{dict_str}}}, domain={self.domain})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def keys(self) -> list[str]:
        """
        Get the field names stored in this object.

        Returns:
            List of string keys in insertion order.

        """
        return list(self.tensor_dict.keys())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Move all tensors to the specified device and dtype.

        Args:
            device: Target device; passed directly to ``torch.Tensor.to``.
            dtype: Target dtype; passed directly to ``torch.Tensor.to``.

        Returns:
            A new instance of the same type with tensors on the requested device/dtype.

        """
        new_tensor_dict = {
            key: tensor.to(device=device, dtype=dtype)
            for key, tensor in self.tensor_dict.items()
        }
        return type(self)(new_tensor_dict, domain=self.domain)

    @classmethod
    def save(cls, data_list: list[EqData], path: str) -> None:
        """
        Concatenate a list of EqData objects and serialize them to disk.

        Args:
            data_list: Non-empty list of EqData objects with matching dtypes,
                devices, shapes, and domains.
            path: Destination ``.pt`` file path.

        Raises:
            AssertionError: If ``data_list`` is empty or the objects are inconsistent.

        """
        assert len(data_list) > 0, "data_list must be a non-empty list"
        assert all(data.dtype == data_list[0].dtype for data in data_list), (
            "All data must have the same dtype."
        )
        assert all(data.device == data_list[0].device for data in data_list), (
            "All data must have the same device."
        )
        assert all(data.shapes == data_list[0].shapes for data in data_list), (
            "All data must have the same shapes."
        )
        assert all(data.domain == data_list[0].domain for data in data_list), (
            "All data must have the same domain."
        )

        dict_to_save = {}
        for key in data_list[0].tensor_dict.keys():
            dict_to_save[key] = torch.cat(
                [data.tensor_dict[key].cpu() for data in data_list], dim=0
            )
        dict_to_save["domain"] = data_list[0].domain

        torch.save(dict_to_save, path)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load an EqData object from a ``.pt`` file written by :meth:`save`.

        Args:
            path: Path to the ``.pt`` file.

        Returns:
            An instance of the calling class with tensors on CPU.

        """
        loaded_dict = torch.load(path, map_location="cpu")
        domain = loaded_dict.pop("domain")
        return cls(loaded_dict, domain=domain)

    @overload
    def __getitem__(self, idx: str) -> torch.Tensor: ...

    @overload
    def __getitem__(self, idx: int | slice | torch.Tensor) -> Self: ...

    def __getitem__(self, idx: int | slice | torch.Tensor | str) -> torch.Tensor | Self:
        """
        Get a subset of the EqData object based on the provided index.

        Args:
            idx: Index, slice or Tensor, or string key to select specific samples
                or a specific tensor from the solution
        Returns:
            EqData object containing the selected samples or a specific tensor

        """
        if isinstance(idx, str):
            return self.tensor_dict[idx]
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        new_tensor_dict = {}
        for key, tensor in self.tensor_dict.items():
            new_tensor_dict[key] = tensor[idx]
        return type(self)(new_tensor_dict, domain=self.domain)

    @classmethod
    def from_list(cls, data_list: list[Self]) -> Self:
        """
        Create a single EqData object by concatenating a list of EqData
        objects.

        Args:
            data_list: List of EqData objects to concatenate
        Returns:
            EqData object containing the concatenated solutions

        """
        assert all(data.domain == data_list[0].domain for data in data_list), (
            "All data must have the same domain."
        )
        assert all(data.dtype == data_list[0].dtype for data in data_list), (
            "All data must have the same dtype."
        )
        assert all(data.device == data_list[0].device for data in data_list), (
            "All data must have the same device."
        )
        assert all(data.shapes == data_list[0].shapes for data in data_list), (
            "All data must have the same shapes."
        )

        new_tensor_dict = {
            key: torch.cat([data.tensor_dict[key] for data in data_list], dim=0)
            for key in data_list[0].tensor_dict
        }
        return cls(new_tensor_dict, domain=data_list[0].domain)

    def new_data(
        self,
        tensor_dict: dict[str, torch.Tensor],
    ) -> Self:
        """
        Create a new EqData object with the same metadata but different solution
        tensor.

        Args:
            tensor_dict: Dictionary containing the solution tensors
        Returns:
            EqData object with the new solution tensor

        """
        new_tensor_dict = self.tensor_dict.copy()
        new_tensor_dict.update(tensor_dict)
        return type(self)(new_tensor_dict, domain=self.domain)


class EqSolution(EqData):
    """
    Data structure to hold the solution of a PDE equation. Supports moving to
    different devices and dtypes.
    """

    def new_solution(self, tensor_dict: dict[str, torch.Tensor]) -> Self:
        """
        Create a new EqSolution sharing the same domain but with updated tensors.

        Args:
            tensor_dict: Replacement tensors; existing keys are overwritten and new
                keys are added.

        Returns:
            A new EqSolution with the merged tensor dict.

        """
        return self.new_data(tensor_dict)


class EqCondition(EqData):
    """
    Data structure to hold the input v = (u(0, x), f(t, x)) of a PDE, named the
    condition in the code. Supports moving to different devices and dtypes.
    """

    def new_condition(self, tensor_dict: dict[str, torch.Tensor]) -> Self:
        """
        Create a new EqCondition sharing the same domain but with updated tensors.

        Args:
            tensor_dict: Replacement tensors; existing keys are overwritten and new
                keys are added.

        Returns:
            A new EqCondition with the merged tensor dict.

        """
        return self.new_data(tensor_dict)
