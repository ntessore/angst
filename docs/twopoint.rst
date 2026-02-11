Two-point functions
===================

.. currentmodule:: angst


Spectra and correlation functions
---------------------------------

.. autofunction:: cl2corr
.. autofunction:: corr2cl

.. autofunction:: cl2var


Shot noise bias
---------------

Compute the value of the shot noise bias contribution to angular power spectra.
For reference, see [arXiv:2408.16903]_ and [arXiv:2507.03749]_.

.. autofunction:: shotnoise


Sets of two-point functions
---------------------------

.. autofunction:: indices2
.. autofunction:: enumerate2


.. _twopoint_order:

Standard order
--------------

All functions that process sets of two-point functions expect them as a
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

In particular, two-point functions for the first :math:`n` fields are contained
in the first :math:`T_n = n \, (n + 1) / 2` entries of the sequence.

To easily generate or iterate over sequences of two-point functions in standard
order, see the :func:`enumerate2` and :func:`indices2` functions.
