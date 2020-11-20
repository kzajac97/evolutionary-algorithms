from typing import Any, Dict

import numpy as np
from scipy import stats
from tqdm import tqdm as progress_bar

from src import operators


def initialize_population(center: float, std: float, size: int, dimensionality: int):
    """
    Initialize population for experiments

    :param center: point around which to generate solutions
    :param std: deviation of random noise used to generate population
    :param size: number of solutions
    :param dimensionality: number of components for each solution

    :return: array of generated solutions
    """
    return (
        np.full(size * dimensionality, center).reshape(size, dimensionality)
        + stats.uniform.rvs(size=size * dimensionality, scale=std).reshape(size, dimensionality)
    )


def optimize(
    objective,
    n_generations: int,
    population_size: int,
    initialize_params: Dict[str, Any],
    mutation_rate: float = 0.2,
    mutation_std: float = 0.2
):
    """
    :param objective: objective function
    :param n_generations: number of generations to run optimization
    :param population_size: number of individual solutions in population
    :param initialize_params: parameters of initializer function
    :param mutation_rate: fraction of population to undergo mutation in each generation
    :param mutation_std: standard deviation in normal distribution used in mutation

    :return: best fitness value in final epoch
    """
    population = initialize_population(**initialize_params)

    for _ in progress_bar(range(n_generations)):
        mating_pool = operators.selection.roulette_selection(population, objective)
        offspring = operators.crossover.vectorized_hypersphere_crossover(mating_pool, n_offspring=population_size)

        to_mutate = np.random.randint(0, population_size, int(mutation_rate * population_size))
        offspring[to_mutate] = operators.mutation.vectorized_gaussian_variable_mutation(offspring[to_mutate], std=mutation_std)

        population = offspring

    return np.apply_along_axis(objective, 1, population).max()
