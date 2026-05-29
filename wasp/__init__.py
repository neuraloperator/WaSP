"""
Main package for the WaSP framework.

This package provides the core functionality for the WaSP framework, including
neural operator training, data generation, and utility functions.
"""

import torch

from . import equations, training, utils

# ensure ieee ft32 precision is used for all computations.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

__all__ = ["equations", "training", "utils"]
