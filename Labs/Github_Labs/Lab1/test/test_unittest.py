import unittest
import os
import sys
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import calculator
from src.area_calculator import (
    area_square,
    area_rectangle,
    area_circle,
    area_triangle,
    area_trapezoid,
)

class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)

        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        self.assertEqual(calculator.fun4(2, 3, 5), 10)
        self.assertEqual(calculator.fun4(5, 0, -1), 4)
        self.assertEqual(calculator.fun4(-1, -1, -1), -3)
        self.assertEqual(calculator.fun4(-1, -1, 100), 98)


if __name__ == '__main__':
    unittest.main()

# Area Calculator Tests


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
