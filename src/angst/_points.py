"""Functions to handle points on the sphere."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def displace(
    lon: ArrayLike, lat: ArrayLike, alpha: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Displace points with longitude *lon* and latitude *lat* (both in
    degrees) by *alpha*.  Inputs must be array-like.  The displacement
    *alpha* must be complex-valued or have a leading axis of size 2 for
    its real and imaginary component.

    Returns a tuple *newlon, newlat* of longitude and latitude (both in
    degrees) of the displaced points.
    """

    alpha = np.asanyarray(alpha)
    if np.iscomplexobj(alpha):
        alpha1, alpha2 = alpha.real, alpha.imag
    else:
        alpha1, alpha2 = alpha

    # we know great-circle navigation:
    # θ' = arctan2(√[(cosθ sin|α| - sinθ cos|α| cosγ)² + (sinθ sinγ)²],
    #              cosθ cos|α| + sinθ sin|α| cosγ)
    # δ = arctan2(sin|α| sinγ, sinθ cos|α| - cosθ sin|α| cosγ)

    t = np.radians(lat)
    ct, st = np.sin(t), np.cos(t)  # sin and cos flipped: lat not co-lat

    a = np.hypot(alpha1, alpha2)  # abs(alpha)
    g = np.arctan2(alpha2, alpha1)  # arg(alpha)
    ca, sa = np.cos(a), np.sin(a)
    cg, sg = np.cos(g), np.sin(g)

    # flipped atan2 arguments for lat instead of co-lat
    tp = np.arctan2(ct * ca + st * sa * cg, np.hypot(ct * sa - st * ca * cg, st * sg))

    d = np.arctan2(sa * sg, st * ca - ct * sa * cg)

    return lon - np.degrees(d), np.degrees(tp)


def displacement(
    from_lon: ArrayLike,
    from_lat: ArrayLike,
    to_lon: ArrayLike,
    to_lat: ArrayLike,
) -> ArrayLike:
    """
    Compute the complex displacement that transforms points with
    longitude *from_lon* and latitude *from_lat* into points with
    longitude *to_lon* and latitude *to_lat* (all in degrees).  All
    inputs must be array-like.
    """

    a = np.radians(np.subtract(90.0, to_lat))
    b = np.radians(np.subtract(90.0, from_lat))
    g = np.radians(np.subtract(from_lon, to_lon))

    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sg, cg = np.sin(g), np.cos(g)

    r = np.arctan2(np.hypot(sa * cb - ca * sb * cg, sb * sg), ca * cb + sa * sb * cg)
    x = sb * ca - cb * sa * cg
    y = sa * sg
    z = np.hypot(x, y)
    return r * (x / z + 1j * y / z)  # type: ignore
