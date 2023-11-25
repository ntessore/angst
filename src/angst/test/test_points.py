import numpy as np
import numpy.testing as npt


def test_displace_arg_complex():
    import angst

    d = 5.0
    r = np.radians(d)

    # north
    lon, lat = angst.displace(0.0, 0.0, r + 0j)
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = angst.displace(0.0, 0.0, -r + 0j)
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = angst.displace(0.0, 0.0, 1j * r)
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = angst.displace(0.0, 0.0, -1j * r)
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_arg_real():
    import angst

    d = 5.0
    r = np.radians(d)

    # north
    lon, lat = angst.displace(0.0, 0.0, [r, 0])
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = angst.displace(0.0, 0.0, [-r, 0])
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = angst.displace(0.0, 0.0, [0, r])
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = angst.displace(0.0, 0.0, [0, -r])
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_abs(rng):
    import angst

    n = 1000
    abs_alpha = rng.uniform(0, 2 * np.pi, size=n)
    arg_alpha = rng.uniform(-np.pi, np.pi, size=n)

    lon_ = np.degrees(rng.uniform(-np.pi, np.pi, size=n))
    lat_ = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))

    lon, lat = angst.displace(lon_, lat_, abs_alpha * np.exp(1j * arg_alpha))

    th = np.radians(90.0 - lat)
    th_ = np.radians(90.0 - lat_)
    delt = np.radians(lon - lon_)

    cos_a = np.cos(th) * np.cos(th_) + np.cos(delt) * np.sin(th) * np.sin(th_)

    npt.assert_allclose(cos_a, np.cos(abs_alpha))


def test_displacement(rng):
    from textwrap import dedent
    import angst

    # unit changes for displacements
    deg5 = np.radians(5.0)
    north = np.exp(1j * 0.0)
    east = np.exp(1j * (np.pi / 2))
    south = np.exp(1j * np.pi)
    west = np.exp(1j * (3 * np.pi / 2))

    # test data: coordinates and expected displacement
    data = [
        # equator
        (0.0, 0.0, 0.0, 5.0, deg5 * north),
        (0.0, 0.0, -5.0, 0.0, deg5 * east),
        (0.0, 0.0, 0.0, -5.0, deg5 * south),
        (0.0, 0.0, 5.0, 0.0, deg5 * west),
        # pole
        (0.0, 90.0, 180.0, 85.0, deg5 * north),
        (0.0, 90.0, -90.0, 85.0, deg5 * east),
        (0.0, 90.0, 0.0, 85.0, deg5 * south),
        (0.0, 90.0, 90.0, 85.0, deg5 * west),
    ]

    # test each displacement individually
    for from_lon, from_lat, to_lon, to_lat, alpha in data:
        alpha_ = angst.displacement(from_lon, from_lat, to_lon, to_lat)
        assert np.allclose(alpha_, alpha), dedent(
            f"""
            displacement from ({from_lon}, {from_lat}) to ({to_lon}, {to_lat})
            distance: expected {np.abs(alpha)}, got {np.abs(alpha_)}
            direction: expected {np.angle(alpha)}, got {np.angle(alpha_)}
            """
        )

    # test on an array
    alpha = angst.displacement(
        rng.uniform(-180.0, 180.0, size=(20, 1)),
        rng.uniform(-90.0, 90.0, size=(20, 1)),
        rng.uniform(-180.0, 180.0, size=5),
        rng.uniform(-90.0, 90.0, size=5),
    )
    assert alpha.shape == (20, 5)
