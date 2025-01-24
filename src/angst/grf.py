"""
Transformations of Gaussian random fields.
"""

from __future__ import annotations

__all__ = [
    "Lognormal",
    "LognormalXNormal",
    "SquaredNormal",
    "Transformation",
    "solve",
]

import math
from dataclasses import dataclass

import numpy as np

# typing
from typing import Any, Callable, Protocol
from numpy.typing import NDArray


class Transformation(Protocol):
    """
    Protocol for transformations of Gaussian random fields.
    """

    def __call__(self, x: NDArray[Any], var: float, /) -> NDArray[Any]:
        """
        Transform a Gaussian correlation function.
        """

    def inv(self, x: NDArray[Any], var: float, /) -> NDArray[Any]:
        """
        Inverse transform to a Gaussian correlation function.
        """

    def der(self, x: NDArray[Any], var: float, /) -> NDArray[Any]:
        """
        Derivative of the transform.
        """


def _relerr(dx: NDArray[Any], x: NDArray[Any]) -> float:
    """compute the relative error max(|dx/x|)"""
    q = np.divide(dx, x, where=(dx != 0), out=np.zeros_like(dx))
    return np.fabs(q).max()  # type: ignore


def solve(
    cl: NDArray[Any],
    tfm: Transformation,
    pad: int = 0,
    *,
    initial: NDArray[Any] | None = None,
    cltol: float = 1e-5,
    gltol: float = 1e-5,
    maxiter: int = 20,
    monopole: float | None = None,
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """
    Solve for a Gaussian angular power spectrum.

    Parameters
    ----------
    cl : (n,) array
    tfm : :class:`Transformation`
    pad : int

    Returns
    -------
    gl : (n,) array
        Gaussian angular power spectrum solution.
    cl : (n + pad,) array
        Realised transformed angular power spectrum.
    info : {0, 1, 2, 3}
        Indicates success of failure of the solution.  Possible values are

        * ``0``, solution did not converge in *maxiter* iterations;
        * ``1``, solution converged in *cl* relative error;
        * ``2``, solution converged in *gl* relative error;
        * ``3``, solution converged in both *cl* and *gl* relative error.

    """

    from ._twopoint import corr2cl, cl2corr, cl2var

    n = len(cl)
    if not isinstance(pad, int) or pad < 0:
        raise TypeError("pad must be a positive integer")

    if initial is None:
        gl = corr2cl(tfm.inv(cl2corr(cl), cl2var(cl)))
    else:
        gl = np.empty(n)
        gl[: len(initial)] = initial[:n]

    if monopole is not None:
        gl[0] = monopole

    gt = cl2corr(np.pad(gl, (0, pad)))
    var = cl2var(gl)
    rl = corr2cl(tfm(gt, var))
    fl = rl[:n] - cl
    if monopole is not None:
        fl[0] = 0
    clerr = _relerr(fl, cl)

    info = 0
    for i in range(maxiter):
        if clerr <= cltol:
            info |= 1
        if info > 0:
            break

        ft = cl2corr(np.pad(fl, (0, pad)))
        dt = tfm.der(gt, var)
        xl = -corr2cl(ft / dt)[:n]
        if monopole is not None:
            xl[0] = 0

        while True:
            gl_ = gl + xl
            gt_ = cl2corr(np.pad(gl_, (0, pad)))
            var_ = cl2var(gl_)
            rl_ = corr2cl(tfm(gt_, var_))
            fl_ = rl_[:n] - cl
            if monopole is not None:
                fl_[0] = 0
            clerr_ = _relerr(fl_, cl)
            if clerr_ <= clerr:
                break
            xl /= 2

        if _relerr(xl, gl) <= gltol:
            info |= 2

        gl, gt, var, rl, fl, clerr = gl_, gt_, var_, rl_, fl_, clerr_

    return gl, rl, info


@dataclass
class Lognormal:
    """
    Transformation for lognormal fields.
    """

    lamda1: float = 1.0
    lamda2: float = 1.0

    def __call__(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        xp = x.__array_namespace__()
        return self.lamda1 * self.lamda2 * xp.expm1(x)  # type: ignore[no-any-return]

    def inv(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        xp = x.__array_namespace__()
        return xp.log1p(x / (self.lamda1 * self.lamda2))  # type: ignore[no-any-return]

    def der(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        xp = x.__array_namespace__()
        return self.lamda1 * self.lamda2 * xp.exp(x)  # type: ignore[no-any-return]


@dataclass
class LognormalXNormal:
    """
    Transformation for cross-correlation between lognormal and Gaussian
    fields.
    """

    lamda: float = 1.0

    def __call__(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        return self.lamda * x

    def inv(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        return x / self.lamda

    def der(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        return self.lamda + (0.0 * x)


@dataclass
class SquaredNormal:
    """
    Squared normal field.  The parameters *a1*, *a2* can be set to
    ``None``, in which case they are inferred from the variance of
    the field.
    """

    a1: float | None = None
    a2: float | None = None
    lamda1: float = 1.0
    lamda2: float = 1.0

    def _pars(self, var: float) -> tuple[float, float]:
        a1 = math.sqrt(1 - var) if self.a1 is None else self.a1
        a2 = math.sqrt(1 - var) if self.a2 is None else self.a2
        return a1 * a2, self.lamda1 * self.lamda2

    def __call__(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        aa, ll = self._pars(var)
        return 2 * ll * x * (x + 2 * aa)

    def inv(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        xp = x.__array_namespace__()
        aa, ll = self._pars(var)
        return xp.sqrt(x / (2 * ll) + aa**2) - aa  # type: ignore[no-any-return]

    def der(self, x: NDArray[Any], var: float) -> NDArray[Any]:
        aa, ll = self._pars(var)
        return 4 * ll * (x + aa)


Alm = NDArray[np.complexfloating[Any, Any]]


def spectrum_from_sht(
    cl: NDArray[Any],
    tfm: Transformation,
    sht: Callable[[NDArray[Any], int], Alm],
    isht: Callable[[Alm], NDArray[Any]],
) -> NDArray[Any]:
    """
    Compute a Gaussian angular power spectrum for the target spectrum
    *cl* and the transformation *tfm*.  Uses the spherical harmonic
    transform pair *sht* and *isht* to compute a simple band-limited
    Gaussian power spectrum, without the full machinery of
    :func:`solve`.
    """

    xp = cl.__array_namespace__()

    # get lmax from cl
    lmax = cl.size - 1

    # store prefactor, compute variance
    fl = (2 * xp.arange(lmax + 1) + 1) / (4 * xp.pi)
    var = (fl * cl).sum()
    fl = xp.sqrt(fl)

    # compute alms for m=0, rest zero
    alm = xp.concat(
        [
            fl * cl,
            xp.zeros(lmax * (lmax + 1) // 2, dtype=complex),
        ],
    )

    # convert to field f(\theta, \phi) = C(\theta) exp(im\phi)
    m = isht(alm)

    # apply lognormal transform to C(theta)
    # (for complex fields, modify absolute value here)
    m = tfm.inv(m, var)

    # get transformed alms
    alm = sht(m, lmax)

    # read Gaussian spectrum from m=0
    return alm.real[: lmax + 1] / fl  # type: ignore[no-any-return]
