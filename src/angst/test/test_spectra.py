import numpy as np
import pytest


def test_enumerate_spectra():
    import angst

    n = 100
    tn = n * (n + 1) // 2

    # create mock spectra with 1 element counting to tn
    spectra = np.arange(tn).reshape(tn, 1)

    # this is the expected order of indices
    indices = [(i, j) for i in range(n) for j in range(i, -1, -1)]

    # iterator that will enumerate the spectra for checking
    it = angst.enumerate_spectra(spectra)

    # go through expected indices and values and compare
    for k, (i, j) in enumerate(indices):
        assert next(it) == (i, j, k)

    # make sure iterator is exhausted
    with pytest.raises(StopIteration):
        next(it)


def test_spectra_indices():
    import angst

    assert list(angst.spectra_indices(0)) == []
    assert list(angst.spectra_indices(1)) == [(0, 0)]
    assert list(angst.spectra_indices(2)) == [(0, 0), (1, 1), (1, 0)]
    assert list(angst.spectra_indices(3)) == [
        (0, 0),
        (1, 1),
        (1, 0),
        (2, 2),
        (2, 1),
        (2, 0),
    ]
