"""
Utilities for the WaSP package.
"""

from .differentiation import FiniteDifference
from .optimal_transport import OTMapping
from .utils import h1_norm, l2_norm, sample_grf

__all__ = [
    "OTMapping",
    "FiniteDifference",
    "l2_norm",
    "h1_norm",
    "sample_grf",
]
