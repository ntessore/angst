import numpy as np
import numpy.testing as npt


def test_displace_arg_complex():
    from angst import displace

    d = 5.0
    r = np.radians(d)

    # north
    lon, lat = displace(0.0, 0.0, r + 0j)
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = displace(0.0, 0.0, -r + 0j)
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = displace(0.0, 0.0, 1j * r)
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = displace(0.0, 0.0, -1j * r)
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_arg_real():
    from angst import displace

    d = 5.0
    r = np.radians(d)

    # north
    lon, lat = displace(0.0, 0.0, [r, 0])
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = displace(0.0, 0.0, [-r, 0])
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = displace(0.0, 0.0, [0, r])
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = displace(0.0, 0.0, [0, -r])
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_abs(rng):
    from angst import displace

    n = 1000
    abs_alpha = rng.uniform(0, 2 * np.pi, size=n)
    arg_alpha = rng.uniform(-np.pi, np.pi, size=n)

    lon_ = np.degrees(rng.uniform(-np.pi, np.pi, size=n))
    lat_ = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))

    lon, lat = displace(lon_, lat_, abs_alpha * np.exp(1j * arg_alpha))

    th = np.radians(90.0 - lat)
    th_ = np.radians(90.0 - lat_)
    delt = np.radians(lon - lon_)

    cos_a = np.cos(th) * np.cos(th_) + np.cos(delt) * np.sin(th) * np.sin(th_)

    npt.assert_allclose(cos_a, np.cos(abs_alpha))
