"""Operations on angular power spectra."""

from __future__ import annotations

import numpy as np

# typing
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Sequence
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from typing import TypeAlias

from . import _core


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

    >>> list(spectra_indices(3))
    [(0, 0), (1, 1), (1, 0), (2, 2), (2, 1), (2, 0)]

    """

    for i in range(n):
        for j in range(i, -1, -1):
            yield i, j


def _cov_clip_negative_eigenvalues(cov: NDArray[Any]) -> NDArray[Any]:
    """
    Covariance matrix from clipping negative eigenvalues.
    """

    # set negative eigenvalues to zero
    w, v = np.linalg.eigh(cov)
    w[w < 0] = 0

    # put matrix back together
    reg: NDArray[Any] = np.einsum("...ij,...j,...kj->...ik", v, w, v)

    # fix the upper triangular part of the matrix to zero
    reg[(...,) + np.triu_indices(w.shape[-1], 1)] = 0

    # return the regularised covariance matrix
    return reg


def _cov_nearest_correlation_matrix(cov: NDArray[Any], niter: int = 20) -> NDArray[Any]:
    """
    Covariance matrix from nearest correlation matrix.  Uses the
    algorithm of Higham (2002).
    """

    # size of the covariance matrix
    s = np.shape(cov)
    n = s[-1]

    # make a copy to work on
    corr = np.copy(cov)

    # view onto the diagonal of the correlation matrix
    diag = corr.reshape(s[:-2] + (-1,))[..., :: n + 1]

    # set correlations with nonpositive diagonal to zero
    good = diag > 0
    corr *= good[..., np.newaxis, :]
    corr *= good[..., :, np.newaxis]

    # get sqrt of the diagonal for normalization
    norm = np.sqrt(diag)

    # compute the correlation matrix
    np.divide(corr, norm[..., np.newaxis, :], where=good[..., np.newaxis, :], out=corr)
    np.divide(corr, norm[..., :, np.newaxis], where=good[..., :, np.newaxis], out=corr)

    # indices of the upper triangular part of the matrix
    triu = (...,) + np.triu_indices(n, 1)

    # always keep upper triangular part of matrix fixed to zero
    # otherwise, Dykstra's correction points in the wrong direction
    corr[triu] = 0

    # find the nearest covariance matrix with given diagonal
    dyks = np.zeros_like(corr)
    proj = np.empty_like(corr)
    for k in range(niter):
        # apply Dykstra's correction to current result
        np.subtract(corr, dyks, out=proj)

        # project onto positive semi-definite matrices
        w, v = np.linalg.eigh(proj)
        w[w < 0] = 0
        np.einsum("...ij,...j,...kj->...ik", v, w, v, out=corr)

        # keep upper triangular part fixed to zero
        corr[triu] = 0

        # compute Dykstra's correction
        np.subtract(corr, proj, out=dyks)

        # project onto matrices with unit diagonal
        diag[good] = 1

    # put the normalisation back to convert correlations to covariance
    np.multiply(corr, norm[..., np.newaxis, :], out=corr)
    np.multiply(corr, norm[..., :, np.newaxis], out=corr)

    # return the regularised covariance matrix
    return corr


# valid ``method`` parameter values for :func:`regularized_spectra`
RegularizedSpectraMethod: TypeAlias = Literal["nearest", "clip"]


def regularized_spectra(
    spectra: Sequence[ArrayLike],
    lmax: int | None = None,
    method: RegularizedSpectraMethod = "nearest",
    **method_kwargs: Any,
) -> list[ArrayLike]:
    """
    Regularises a complete set *spectra* of angular power spectra in
    :ref:`standard order <spectra_order>` such that at every angular
    mode number :math:`\\ell`, the matrix :math:`C_\\ell^{ij}` is a
    valid positive semi-definite covariance matrix.

    The length of the returned spectra is set by *lmax*, or the maximum
    length of the given spectra if *lmax* is not given.  Shorter input
    spectra are padded with zeros as necessary.  Missing spectra can be
    set to :data:`None` or, preferably, an empty array.

    The *method* parameter determines how the regularisation is carried
    out.  The following methods are supported:

    ``regularized_spectra(..., method="nearest", niter=20)``
        Compute the (possibly defective) correlation matrices of the
        given spectra, then find the nearest valid correlation matrices,
        using the algorithm from [Higham02]_ for *niter* iterations.
        This keeps the diagonals (i.e. auto-correlations) fixed, but
        requires all of them to be nonnegative.

    ``regularized_spectra(..., method="clip")``
        Compute the eigendecomposition of the given spectra's matrices
        and set all negative eigenvalues to zero.

    """

    # regularise the cov matrix using the chosen method
    if method == "clip":
        method_func = _cov_clip_negative_eigenvalues
    elif method == "nearest":
        method_func = _cov_nearest_correlation_matrix
    else:
        raise ValueError(f"unknown method '{method}'")

    # recover the number of fields from the number of spectra
    try:
        n = _core.inv_triangle_number(len(spectra))
    except ValueError as exc:
        raise ValueError("invalid number of spectra") from exc

    if lmax is None:
        # maximum length in input spectra
        k = max((np.size(cl) for cl in spectra if cl is not None), default=0)
    else:
        k = lmax + 1

    # this is the covariance matrix of the spectra
    # the leading dimension is k, then it is a n-by-n covariance matrix
    # missing entries are zero, which is the default value
    cov = np.zeros((k, n, n))

    # fill the matrix up by going through the spectra in order
    # skip over entries that are None
    # if the spectra are ragged, some entries at high ell may remain zero
    # only fill the lower triangular part, everything is symmetric
    for i, j, cl in enumerate_spectra(spectra):
        if cl is not None:
            cov[: np.size(cl), j, i] = np.reshape(cl, -1)[:k]

    # use cholesky() as a fast way to check for positive semi-definite
    # if it fails, the matrix of spectra needs regularisation
    # otherwise, the matrix is pos. def. and the spectra are good
    try:
        np.linalg.cholesky(cov + np.finfo(0.0).tiny)
    except np.linalg.LinAlgError:
        # regularise the cov matrix using the chosen method
        cov = method_func(cov, **method_kwargs)

    # gather regularised spectra from array
    # convert matrix slices to contiguous arrays for type safety
    reg: list[ArrayLike] = []
    for i, j in spectra_indices(n):
        reg.append(np.ascontiguousarray(cov[:, j, i]))

    # return the regularised spectra
    return reg
