Angular power spectra
=====================

.. currentmodule:: angst


Sets of angular power spectra
-----------------------------

.. autofunction:: spectra_indices
.. autofunction:: enumerate_spectra


.. _spectra_order:

Standard order
--------------

All functions that process sets of angular power spectra expect them as a
sequence using the following "Christmas tree" ordering:

.. raw:: html
   :file: figures/spectra_order.svg

In other words, the sequence begins as such:

* Index 0 describes the auto-correlation of field 0,
* index 1 describes the auto-correlation of field 1,
* index 2 describes the cross-correlation of field 1 and field 0,
* index 3 describes the auto-correlation of field 2,
* index 4 describes the cross-correlation of field 2 and field 1,
* index 5 describes the cross-correlation of field 2 and field 0,
* etc.

In particular, cross-correlations for the first :math:`n` fields are contained
in the first :math:`T_n = n \, (n + 1) / 2` entries of the sequence.

To easily generate or iterate over sequences of angular power spectra in
standard order, see the :func:`enumerate_spectra` and :func:`spectra_indices`
functions.
