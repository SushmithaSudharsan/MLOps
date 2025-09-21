import math


def area_square(side: float) -> float:
    if side < 0:
        raise ValueError("Side must be non-negative")
    return side * side


def area_rectangle(length: float, width: float) -> float:
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width


def area_circle(radius: float) -> float:
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius ** 2


def area_triangle(base: float, height: float) -> float:
    if base < 0 or height < 0:
        raise ValueError("Base and height must be non-negative")
    return 0.5 * base * height


def area_trapezoid(base1: float, base2: float, height: float) -> float:
    if base1 < 0 or base2 < 0 or height < 0:
        raise ValueError("All values must be non-negative")
    return 0.5 * (base1 + base2) * height
