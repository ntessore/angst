"""Operations on angular power spectra."""

from __future__ import annotations

# typing
from typing import Iterable, Iterator
from numpy.typing import ArrayLike


def enumerate_spectra(
    spectra: Iterable[ArrayLike | None],
) -> Iterator[tuple[int, int, ArrayLike | None]]:
    """
    Iterate over a set of angular power spectra in :ref:`standard order
    <spectra_order>`, returning a tuple of indices and their associated
    spectrum from the input.

    >>> spectra = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> list(enumerate_spectra(spectra))
    [(0, 0, [1, 2, 3]), (1, 1, [4, 5, 6]), (1, 0, [7, 8, 9])]

    """

    for k, cl in enumerate(spectra):
        i = int((2 * k + 0.25) ** 0.5 - 0.5)
        j = i * (i + 3) // 2 - k
        yield i, j, cl


def spectra_indices(n: int) -> Iterator[tuple[int, int]]:
    """
    Return an iterator over indices in :ref:`standard order
    <spectra_order>` for a set of spectra for *n* functions.  Each item
    is a tuple of indices *i*, *j*.

    >>> list(spectrum_indices(3))
    [(0, 0), (1, 1), (1, 0), (2, 2), (2, 1), (2, 0)]

    """

    for i in range(n):
        for j in range(i, -1, -1):
            yield i, j
