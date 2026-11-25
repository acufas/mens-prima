from math import tanh as _tanh
from math import exp

def relu(x: float) -> float:
    """Return the rectified linear activation of x."""
    return x if x > 0 else 0.0


def identity(x: float) -> float:
    """Return x unchanged."""
    return x


def sigmoid(x: float) -> float:
    """Return the logistic sigmoid of x, in the range (0, 1)."""
    if x > 0:
        denominator = 1 + exp(-x)
        return 1 / denominator
    e_to_pow = exp(x)
    denominator = 1 + e_to_pow
    return e_to_pow / denominator


def tanh(x: float) -> float:
    """Return the hyperbolic tangent of x, in the range (-1, 1)."""
    return _tanh(x)


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Return x if x is positive, otherwise alpha times x."""
    return x if x > 0 else (alpha * x)

