import pytest


def test_inv_triangle_number():
    from angst import inv_triangle_number

    for n in range(10_000):
        assert inv_triangle_number(n * (n + 1) // 2) == n

    # check floats
    assert inv_triangle_number(0.0) == 0
    assert inv_triangle_number(1.0) == 1
    assert inv_triangle_number(3.0) == 2

    for t in 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20:
        with pytest.raises(ValueError):
            inv_triangle_number(t)
