import numpy as np


def row_wise_multiply(x: np.array, y: np.array) -> np.array:
    """
    Perform row wise multiplication

    Example:
       >>> row_wise_multiply([1, -1], [[-10, 10], [-10, 10]])
       ... [[-10, 10], [10, -10]]
    """
    return np.apply_along_axis(np.multiply, 0, x, y)
