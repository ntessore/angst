"""
Module for two-point functions.
"""

from __future__ import annotations

import math

import numpy as np

# typing
from typing import Any, Iterable, Iterator
from numpy.typing import ArrayLike, NDArray


def enumerate2(
    entries: Iterable[ArrayLike | None],
) -> Iterator[tuple[int, int, ArrayLike | None]]:
    """
    Iterate over a set of two-point functions in :ref:`standard order
    <twopoint_order>`, returning a tuple of indices and their associated entry
    from the input.

    >>> spectra = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> list(enumerate2(spectra))
    [(0, 0, [1, 2, 3]), (1, 1, [4, 5, 6]), (1, 0, [7, 8, 9])]

    """

    for k, cl in enumerate(entries):
        i = int((2 * k + 0.25) ** 0.5 - 0.5)
        j = i * (i + 3) // 2 - k
        yield i, j, cl


def indices2(n: int) -> Iterator[tuple[int, int]]:
    """
    Return an iterator over indices in :ref:`standard order <twopoint_order>`
    for a set of two-point functions for *n* fields.  Each item is a tuple of
    indices *i*, *j*.

    >>> list(indices2(3))
    [(0, 0), (1, 1), (1, 0), (2, 2), (2, 1), (2, 0)]

    """

    for i in range(n):
        for j in range(i, -1, -1):
            yield i, j


def cl2corr(cl: NDArray[Any], closed: bool = False) -> NDArray[Any]:
    r"""transform angular power spectrum to correlation function

    Takes an angular power spectrum with :math:`\mathtt{n} = \mathtt{lmax}+1`
    coefficients and returns the corresponding angular correlation function in
    :math:`\mathtt{n}` points.

    The correlation function values can be computed either over the closed
    interval :math:`[0, \pi]`, in which case :math:`\theta_0 = 0` and
    :math:`\theta_{n-1} = \pi`, or over the open interval :math:`(0, \pi)`.

    Parameters
    ----------
    cl : (n,) array_like
        Angular power spectrum from :math:`0` to :math:`\mathtt{lmax}`.
    closed : bool
        Compute correlation function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    corr : (n,) array_like
        Angular correlation function.

    """

    from flt import idlt  # type: ignore [import-not-found]

    # length n of the transform
    if cl.ndim != 1:
        raise TypeError("cl must be 1d array")
    n = cl.shape[-1]

    # DLT coefficients = (2l+1)/(4pi) * Cl
    c = np.arange(1, 2 * n + 1, 2, dtype=float)
    c /= 4 * np.pi
    c *= cl

    # perform the inverse DLT
    corr: NDArray[Any] = idlt(c, closed=closed)

    # done
    return corr


def corr2cl(corr: NDArray[Any], closed: bool = False) -> NDArray[Any]:
    r"""transform angular correlation function to power spectrum

    Takes an angular function in :math:`\mathtt{n}` points and returns the
    corresponding angular power spectrum from :math:`0` to :math:`\mathtt{lmax}
    = \mathtt{n}-1`.

    The correlation function must be given at the angles returned by
    :func:`transformcl.theta`.  These can be distributed either over the closed
    interval :math:`[0, \pi]`, in which case :math:`\theta_0 = 0` and
    :math:`\theta_{n-1} = \pi`, or over the open interval :math:`(0, \pi)`.

    Parameters
    ----------
    corr : (n,) array_like
        Angular correlation function.
    closed : bool
        Compute correlation function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    cl : (n,) array_like
        Angular power spectrum from :math:`0` to :math:`\mathtt{lmax}`.

    """

    from flt import dlt

    # length n of the transform
    if corr.ndim != 1:
        raise TypeError("corr must be 1d array")
    n = corr.shape[-1]

    # compute the DLT coefficients
    cl: NDArray[Any] = dlt(corr, closed=closed)

    # DLT coefficients = (2l+1)/(4pi) * Cl
    cl /= np.arange(1, 2 * n + 1, 2, dtype=float)
    cl *= 4 * np.pi

    # done
    return cl


def cl2var(cl: NDArray[Any]) -> float:
    """
    Compute the variance of the spherical random field in a point from the
    given angular power spectrum.  The input can be multidimensional, with
    the last axis representing the modes.
    """
    ell = np.arange(np.shape(cl)[-1])
    return np.sum((2 * ell + 1) / (4 * np.pi) * cl)  # type: ignore


def shotnoise(
    *,
    values: NDArray[Any] | None = None,
    weights: NDArray[Any] | None = None,
    area: float | None = None,
    nside: int | None = None,
) -> float:
    """
    Compute the shot noise bias from *values* and *weights*.

    The returned value can be normalised by the effective pixel area.

    * If *area* is given, it is assumed that the spectra are computed with a
      convolution kernel of this area.
    * If *nside* is given, it is assumed that the spectra are computed from
      HEALPix maps of this resolution, which implicitly sets *area*.

    This function computes the "raw" shot noise bias of a random field.  The
    returned value should be divided by two to obtain the shot noise bias for
    E/B-mode power spectra.

    """
    # needs one input at least
    if values is None and weights is None:
        raise ValueError("requires values or weights")

    # cannot set nside and area at the same time
    if area is not None and nside is not None:
        raise ValueError("cannot set both area and nside")

    # account for weights
    if weights is not None:
        if values is None:
            values = weights
        else:
            values = weights * values

    assert values is not None

    # compute area from HEALPix NSIDE if given
    if nside is not None:
        area = (4 * math.pi) / (12 * nside**2)

    # gather all prefactors
    fact = 1 / (4 * math.pi)
    if area is not None:
        fact = fact * area**2

    # compute compensated sum from catalgue
    return fact * math.fsum(values.real**2 + values.imag**2)
