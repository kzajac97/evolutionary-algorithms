import numpy as np


def saddle(x, r: float = 1.0) -> float:
    """
    Objective function with saddle shape for any number of dimensions

    :param x: solution vector
    :param r: radius parameter, controls distance between saddles

    :return: objective value for given data point
    """
    return (
        np.exp(-5 * np.sum(np.power(x, 2)))
        + 2 * np.exp(-5 * np.sum(np.power(x - r, 2)))
    )
