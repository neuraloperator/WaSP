"""
Equations module for the WaSP framework.
This module contains the implementation of the Allen-Cahn equation,
the Kuramoto-Sivashinsky equation, and the Complex Ginzburg-Landau equation.
"""

from .ac_equation import ACEquation
from .equation_base import EquationBase
from .equation_data import EqCondition, EqData, EqSolution
from .gl_equation import GLEquation
from .ks_equation import KSEquation

__all__ = [
    "EqData",
    "EqCondition",
    "EqSolution",
    "ACEquation",
    "KSEquation",
    "GLEquation",
    "EquationBase",
]
