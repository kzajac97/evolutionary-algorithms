import numpy as np
from scipy import stats


def vectorized_gaussian_variable_mutation(population: np.array, std: float = 1.0) -> np.array:
    """
    Performs Gaussian mutation operation on population of individuals in floating point gene representation,
    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param population: array of solutions, with shape (N_INDIVIDUALS, N_OBJECTIVE_DIMS, )
    :param std: standard deviation of normal distribution used mutation

    :return: array containing mutated population
    """
    return population + stats.norm.rvs(size=np.product(population.shape), scale=std).reshape(population.shape)


def gaussian_standard_mutation(individual: np.array, mutation_strength: float = 0.1) -> np.array:
    """
    Performs Gaussian mutation operation on single individual in floating point gene representation,
    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param individual: single solution represented as numpy array with shape (N_OBJECTIVE_DIMS, )
    :param mutation_strength: strength of mutation operation, should be adjusted based on search space size

    :return: Mutated individual solution
    """
    offset = np.random.randn(individual.shape[0])
    return mutation_strength * offset + individual


def vectorized_gaussian_standard_mutation(population: np.array, mutation_strength: float = 0.1) -> np.array:
    """
    Performs Gaussian mutation operation on population of individuals in floating point gene representation,
    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param population: array of solutions, with shape (N_INDIVIDUALS, N_OBJECTIVE_DIMS, )
    :param mutation_strength: strength of mutation operation, should be adjusted based on search space size

    :return: array containing mutated population
    """
    offset = np.random.randn(population.shape[0] * population.shape[1]).reshape(population.shape)
    return mutation_strength * offset + population
