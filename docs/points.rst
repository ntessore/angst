Points on the sphere
====================

.. currentmodule:: angst


.. _displacements:

Displacements
-------------

The displacement of a point on the sphere is described by movement along a
geodesic.  In the language of ships, this is called "great-circle navigation",
while in differential geometry, it is the so-called "exponential map".

On the sphere, we encode displacement as a complex value :math:`\alpha` with
spin weight :math:`1`.  The absolute value :math:`|\alpha|` is the angular
distance of the displacement:  If :math:`\hat{u}` and :math:`\hat{u}'` are two
points on the sphere, and :math:`\alpha` is the complex displacement between
them, then :math:`\hat{u} \cdot \hat{u}' = \cos|\alpha|`.

The complex argument :math:`\arg\alpha` (in radians) is the direction of the
displacement: By convention, :math:`\arg\alpha = 0` is north, :math:`\arg\alpha
= \frac{\pi}{2}` is east, :math:`\arg\alpha = \pi` is south, and
:math:`\arg\alpha = -\frac{\pi}{2}` is west.  With this definition, the
gradients of potential fields recover their intuitive meaning, e.g. in
gravitational lensing.

There are two functions for working with points and displacements:  Given a set
of points and their displacements, the :func:`displace` function computes the
new set of points after applying the displacement.  Conversely, given an
initial and a final set of points, the :func:`displacement` function computes
the displacement that would transform the one set into the other.

.. autofunction:: displace
.. autofunction:: displacement
