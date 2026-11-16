"""Small shared validators for experiment contracts; no implicit type coercion."""
from numbers import Integral, Real
import math


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " must be an integer in the supported range.")
    return int(value)


def finite_number(value, name, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(name + " must be finite and numeric.")
    if minimum is not None and value < minimum or strictly_positive and value <= 0:
        raise ValueError(name + " is outside the supported range.")
    return float(value)
