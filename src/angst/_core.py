"""Core functionality that is used in other modules.

This module, and only this module, can be imported by other modules in
this package.

"""

from __future__ import annotations

__all__ = [
    "inv_triangle_number",
]

import math


def inv_triangle_number(triangle_number: int | float) -> int:
    """
    The :math:`n`-th triangle number is :math:`T_n = n \\, (n+1)/2`.  If
    the argument is :math:`T_n`, then :math:`n` is returned.  Otherwise,
    a :class:`ValueError` is raised.
    """

    n: int = math.floor((2 * triangle_number) ** 0.5)
    if n * (n + 1) // 2 != triangle_number:
        raise ValueError(f"not a triangle number: {triangle_number}")
    return n
