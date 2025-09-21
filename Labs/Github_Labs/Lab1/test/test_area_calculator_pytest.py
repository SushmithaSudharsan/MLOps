import math
import pytest
from src.area_calculator import (
    area_square,
    area_rectangle,
    area_circle,
    area_triangle,
    area_trapezoid,
)


def test_area_square():
    assert area_square(4) == 16
    assert area_square(0) == 0
    with pytest.raises(ValueError):
        area_square(-1)


def test_area_rectangle():
    assert area_rectangle(5, 3) == 15
    with pytest.raises(ValueError):
        area_rectangle(-2, 3)


def test_area_circle():
    assert math.isclose(area_circle(1), math.pi, rel_tol=1e-5)
    with pytest.raises(ValueError):
        area_circle(-3)


def test_area_triangle():
    assert area_triangle(10, 5) == 25
    with pytest.raises(ValueError):
        area_triangle(0, -2)


def test_area_trapezoid():
    assert area_trapezoid(3, 5, 4) == 16
    with pytest.raises(ValueError):
        area_trapezoid(-1, 2, 3)
