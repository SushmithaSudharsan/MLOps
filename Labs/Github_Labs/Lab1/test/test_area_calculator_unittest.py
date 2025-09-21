import math
import unittest
from src.area_calculator import (
    area_square,
    area_rectangle,
    area_circle,
    area_triangle,
    area_trapezoid,
)


class TestAreaCalculator(unittest.TestCase):

    def test_area_square(self):
        self.assertEqual(area_square(4), 16)
        self.assertEqual(area_square(0), 0)
        with self.assertRaises(ValueError):
            area_square(-2)

    def test_area_rectangle(self):
        self.assertEqual(area_rectangle(5, 3), 15)
        with self.assertRaises(ValueError):
            area_rectangle(-1, 2)

    def test_area_circle(self):
        self.assertAlmostEqual(area_circle(1), math.pi, places=5)
        with self.assertRaises(ValueError):
            area_circle(-1)

    def test_area_triangle(self):
        self.assertEqual(area_triangle(10, 5), 25)
        with self.assertRaises(ValueError):
            area_triangle(-3, 4)

    def test_area_trapezoid(self):
        self.assertEqual(area_trapezoid(3, 5, 4), 16)
        with self.assertRaises(ValueError):
            area_trapezoid(2, -5, 3)


if __name__ == "__main__":
    unittest.main()
