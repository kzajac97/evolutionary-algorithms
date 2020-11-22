from typing import Any, Dict

import numpy as np
import pandas as pd
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
    mutation_std: float = 0.2,
    silent: bool = True,
    save: bool = True,
    logging_path: str = "./logs.csv",
):
    """
    :param objective: objective function
    :param n_generations: number of generations to run optimization
    :param population_size: number of individual solutions in population
    :param initialize_params: parameters of initializer function
    :param mutation_rate: fraction of population to undergo mutation in each generation
    :param mutation_std: standard deviation in normal distribution used in mutation
    :param silent: if False print progress bar
    :param save: if True save all solutions to file
    :param logging_path: path where solutions are saved

    :return: best fitness value in final epoch
    """
    population = initialize_population(**initialize_params)
    logs = np.zeros([population_size * n_generations, 4])

    for generation in progress_bar(range(n_generations), disable=silent):
        mating_pool = operators.selection.roulette_selection(population, objective)
        offspring = operators.crossover.vectorized_hypersphere_crossover(mating_pool, n_offspring=population_size)

        to_mutate = np.random.randint(0, population_size, int(mutation_rate * population_size))
        offspring[to_mutate] = operators.mutation.vectorized_gaussian_variable_mutation(offspring[to_mutate], std=mutation_std)

        population = offspring
        if save:
            start = generation*population_size
            stop = generation*population_size + population_size
            logs[start:stop, :2] = population
            logs[start:stop, 2] = np.apply_along_axis(objective, 1, population)
            logs[start: stop, 3] = [int(generation)] * population_size

    if save:  # save logs to csv file
        pd.DataFrame.from_records(logs, columns=["x", "y", "value", "generation"]).to_csv(logging_path, index=False)

    return np.apply_along_axis(objective, 1, population).max()
