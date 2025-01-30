"""
Transformations of Gaussian random fields.
"""

from __future__ import annotations

__all__ = [
    "Lognormal",
    "Normal",
    "SquaredNormal",
    "Transformation",
    "compute",
    "solve",
    "corr",
    "icorr",
    "dcorr",
]

import functools
from dataclasses import dataclass

from array_api_compat import array_namespace  # type: ignore[import-not-found]
import flt  # type: ignore[import-not-found]
import numpy as np

# typing
from typing import Any, Callable, Protocol, TypeVar
from numpy.typing import NDArray

ArrayT = TypeVar("ArrayT", bound="Array")
TransformationT = TypeVar("TransformationT", bound="Transformation")


class Array(Protocol):
    """
    Protocol for arrays.
    """

    shape: tuple[int, ...]

    def __add__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __sub__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __mul__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __truediv__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __pow__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...

    def __radd__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rsub__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rmul__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rtruediv__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rpow__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...


class Transformation(Protocol):
    """
    Protocol for transformations of Gaussian random fields.
    """

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        """
        Transform a Gaussian random field *x* with variance *var*.
        """


class Dispatch(Protocol):
    """
    Protocol for the result of dispatch().
    """

    def __call__(
        self, t1: Transformation, t2: Transformation, x: ArrayT, /
    ) -> ArrayT: ...

    def add(
        self,
        impl: Callable[[TransformationT, TransformationT, ArrayT], ArrayT],
    ) -> Callable[[TransformationT, TransformationT, ArrayT], ArrayT]: ...


def dispatch(
    func: Callable[[Transformation, Transformation, ArrayT], ArrayT],
) -> Dispatch:
    """
    Create a simple dispatcher for transformation pairs.
    """

    outer = functools.singledispatch(func)
    dispatch = outer.dispatch
    register = outer.register

    def add(
        impl: Callable[[TransformationT, TransformationT, ArrayT], ArrayT],
    ) -> Callable[[TransformationT, TransformationT, ArrayT], ArrayT]:
        from inspect import signature
        from typing import get_type_hints

        sig = signature(impl)
        if len(sig.parameters) != 3:
            raise TypeError("invalid signature")
        par1, par2, _ = sig.parameters.values()
        if par1.annotation is par1.empty or par2.annotation is par2.empty:
            raise TypeError("invalid signature")
        a, b, *_ = get_type_hints(impl).values()

        inner_a = dispatch(a)
        inner_b = dispatch(b)

        if inner_a is func:
            inner_a = register(a, functools.singledispatch(func))
        if inner_b is func:
            inner_b = register(b, functools.singledispatch(func))

        inner_a.register(b, impl)  # type: ignore [attr-defined]
        inner_b.register(a, lambda t2, t1, x: impl(t1, t2, x))  # type: ignore [attr-defined]

        return impl

    @functools.wraps(func)
    def wrapper(t1: Transformation, t2: Transformation, x: ArrayT) -> ArrayT:
        inner = dispatch(type(t1))
        if inner is not func:
            impl = inner.dispatch(type(t2))  # type: ignore [attr-defined]
        else:
            impl = func
        return impl(t1, t2, x)  # type: ignore [no-any-return]

    wrapper.add = add  # type: ignore [attr-defined]
    return wrapper  # type: ignore [return-value]


@dispatch
def corr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Transform a Gaussian angular correlation function.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The Gaussian angular correlation function.

    Returns
    -------
    y : array_like
        The transformed angular correlation function.

    """

    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def icorr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Inverse-transform an angular correlation function.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The transformed angular correlation function.

    Returns
    -------
    y : array_like
        The Gaussian angular correlation function.

    """

    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def dcorr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Derivative of the angular correlation function transform.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The Gaussian angular correlation function.

    Returns
    -------
    y : array_like
        The derivative of the transformed angular correlation function.

    """

    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def compute(
    cl: ArrayT,
    t1: Transformation,
    t2: Transformation | None = None,
) -> ArrayT:
    """
    Compute a band-limited Gaussian angular power spectrum for the
    target spectrum *cl* and the transformations *t1* and *t2*.  If *t2*
    is not given, it is assumed to be the same as *t1*.

    Parameters
    ----------
    cl : array_like
        The angular power spectrum after the transformations.
    t1, t2 : :class:`Transformation`
        Transformations applied to the Gaussian random field(s).

    Returns
    -------
    gl : array_like
        Gaussian angular power spectrum.

    Examples
    --------
    Compute a Gaussian angular power spectrum ``gl`` for a lognormal
    transformation::

        t = angst.grf.Lognormal()
        gl = angst.grf.compute(cl, t)

    See also
    --------
    angst.grf.solve: Iterative solver for non-band-limited spectra.

    """

    if t2 is None:
        t2 = t1

    xp = array_namespace(cl)

    # get lmax from cl
    lmax = cl.shape[-1] - 1

    # store prefactor
    fl = (2 * xp.arange(lmax + 1) + 1) / (4 * xp.pi)

    # transform C_l to C(\theta), apply transformation, and transform back
    return flt.dlt(icorr(t1, t2, flt.idlt(cl * fl))) / fl  # type: ignore[no-any-return]


def _relerr(dx: NDArray[Any], x: NDArray[Any]) -> float:
    """compute the relative error max(|dx/x|)"""
    q = np.divide(dx, x, where=(dx != 0), out=np.zeros_like(dx))
    return np.fabs(q).max()  # type: ignore


def solve(
    cl: NDArray[Any],
    t1: Transformation,
    t2: Transformation | None = None,
    *,
    pad: int = 0,
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
    t1, t2 : :class:`Transformation`

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

    See also
    --------
    angst.grf.compute: Direct computation for band-limited spectra.

    """

    from ._twopoint import corr2cl, cl2corr

    if t2 is None:
        t2 = t1

    n = len(cl)
    if not isinstance(pad, int) or pad < 0:
        raise TypeError("pad must be a positive integer")

    if initial is None:
        gl = corr2cl(icorr(t1, t2, cl2corr(cl)))
    else:
        gl = np.zeros(n)
        gl[: len(initial)] = initial[:n]

    if monopole is not None:
        gl[0] = monopole

    gt = cl2corr(np.pad(gl, (0, pad)))
    rl = corr2cl(corr(t1, t2, gt))
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
        dt = dcorr(t1, t2, gt)
        xl = -corr2cl(ft / dt)[:n]
        if monopole is not None:
            xl[0] = 0

        while True:
            gl_ = gl + xl
            gt_ = cl2corr(np.pad(gl_, (0, pad)))
            rl_ = corr2cl(corr(t1, t2, gt_))
            fl_ = rl_[:n] - cl
            if monopole is not None:
                fl_[0] = 0
            clerr_ = _relerr(fl_, cl)
            if clerr_ <= clerr:
                break
            xl /= 2

        if _relerr(xl, gl) <= gltol:
            info |= 2

        gl, gt, rl, fl, clerr = gl_, gt_, rl_, fl_, clerr_

    return gl, rl, info


########################################################################
# transformations
########################################################################


@dataclass
class Normal:
    """
    Transformation for normal fields.
    """

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        return x


@dataclass
class Lognormal:
    """
    Transformation for lognormal fields.
    """

    lamda: float = 1.0

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        xp = array_namespace(x)
        x = xp.expm1(x - var / 2)
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


@dataclass
class SquaredNormal:
    """
    Squared normal field.
    """

    a: float
    lamda: float = 1.0

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        x = (x - self.a) ** 2 - 1
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


########################################################################
# normal x normal
########################################################################


@corr.add
def _(t1: Normal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return x


@icorr.add
def _(t1: Normal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return x


@dcorr.add
def _(t1: Normal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return 1.0 + (0 * x)


########################################################################
# lognormal x lognormal
########################################################################


@corr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = array_namespace(x)
    return t1.lamda * t2.lamda * xp.expm1(x)  # type: ignore[no-any-return]


@icorr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = array_namespace(x)
    return xp.log1p(x / (t1.lamda * t2.lamda))  # type: ignore[no-any-return]


@dcorr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = array_namespace(x)
    return t1.lamda * t2.lamda * xp.exp(x)  # type: ignore[no-any-return]


########################################################################
# lognormal x normal
########################################################################


@corr.add
def _(t1: Lognormal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return t1.lamda * x


@icorr.add
def _(t1: Lognormal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return x / t1.lamda


@dcorr.add
def _(t1: Lognormal, t2: Normal, x: ArrayT, /) -> ArrayT:
    return t1.lamda + (0.0 * x)


########################################################################
# squared normal x squared normal
########################################################################


@corr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 2 * ll * x * (x + 2 * aa)


@icorr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    xp = array_namespace(x)
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return xp.sqrt(x / (2 * ll) + aa**2) - aa  # type: ignore[no-any-return]


@dcorr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 4 * ll * (x + aa)
