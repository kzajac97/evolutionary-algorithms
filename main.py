from tqdm import tqdm as progress_bar

from src.genetic import optimize
from src.plot import plot_optimization_history
from src.objectives import saddle


if __name__ == "__main__":
    population_size = 10  # number of solutions
    n_generations = 10_000  # number of steps in optimization
    mutation_rate = 0.8  # when 1.0 all solutions are mutated
    mutation_std = 0.3
    # initial population
    center = 0.0
    deviation = 0.2
    dimensionality = 2  # must be the same as objective
    # number of times experiment is repeated
    n_runs = 1
    save_frequency = 100
    logging_path = r"./logs/solution_logs_at_run_{}.csv"

    for run_id in progress_bar(range(n_runs)):
        save = not bool(run_id % save_frequency)
        fittest = optimize(
            objective=saddle,
            n_generations=n_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_std=mutation_std,
            initialize_params=dict(center=center, std=deviation, size=population_size, dimensionality=dimensionality),
            logging_path=logging_path.format(run_id),
            save=save
        )

    # shows first optimization run, to change insert different value to .format
    plot_optimization_history(logging_path.format(0))
