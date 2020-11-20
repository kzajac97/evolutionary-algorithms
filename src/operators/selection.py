from typing import Callable

import numpy as np
from sklearn.preprocessing import normalize


def roulette_selection(
    population: np.array, objective: Callable[[np.array], float]
) -> np.array:
    """
    Select mating pool using Roulette selection

    :param population: array of individual specimen
    :param objective: objective function, must take in array and return fitness value

    :return: array of selected mating pairs with shape (POPULATION_SIZE, 2)
    """
    fitness = np.apply_along_axis(objective, 1, population)
    probabilities = np.cumsum(normalize(fitness.reshape(-1, 1), norm="l1"))

    selected = np.vectorize(lambda value: len(np.where(value > probabilities)[0]))(np.random.rand(len(population * 2)))
    return population[selected].reshape(len(population), 2)
