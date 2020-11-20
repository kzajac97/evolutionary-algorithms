import numpy as np
from tqdm import tqdm as progress_bar

from src.genetic import optimize
from src.objectives import saddle


if __name__ == "__main__":
    population_size = 10  # number of solutions
    n_generations = 10_000  # number of steps in optimization
    mutation_rate = 0.2
    mutation_std = 1.0
    # initial population
    center = 0.0
    deviation = 0.2
    dimensionality = 2  # must be the same as objective

    fittest = optimize(
        objective=saddle,
        n_generations=n_generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_std=mutation_std,
        initialize_params=dict(center=center, std=deviation, size=population_size, dimensionality=dimensionality),
    )

    print(fittest)
