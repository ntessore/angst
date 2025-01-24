:mod:`angst.grf` --- Gaussian random fields
===========================================

.. currentmodule:: angst.grf
.. module:: angst.grf


Gaussian angular power spectra
------------------------------

.. autofunction:: solve

.. autofunction:: compute_generic


Transformations
---------------

.. class:: Transformation(Protocol)

   .. automethod:: __call__
   .. automethod:: inv
   .. automethod:: der


.. class:: Lognormal

   Implements the :class:`Transformation` for lognormal fields.


.. class:: LognormalXNormal

   Implements the :class:`Transformation` for the cross-correlation between
   :class:`Lognormal` and Gaussian fields.


.. class:: SquaredNormal

   Implements the :class:`Transformation` for squared normal fields.
