from optimizers.ARS import AdaptiveRandomSearch
from optimizers.CEM import CrossEntropyMethod
from optimizers.basin_hopping import BasinHopping
from problems.ackley import AckleyProblem
from problems.rastrigin import RastriginProblem
from problems.knapsack import KnapsackProblem

from optimizers.random_search import RandomSearch
from optimizers.simulated_annealing import SimulatedAnnealing

from experiment.benchmark import Benchmark
from problems.rosenbrock import RosenbrockProblem
from problems.schwefel import SchwefelProblem
from utils import generate_knapsack_instance
from visualization.convergence_plot import plot_convergence


if __name__ == "__main__":
    # ПАРАМЕТРЫ
    weights, values, capacity = generate_knapsack_instance(
        n_items=20,
        capacity_ratio=0.45,
        seed=42
    )
    rastrigin_dimensions  = 2
    rosenbrock_dimensions = 2
    ackley_dimensions     = 2
    schwefel_dimensions   = 2

    iteration_num = 40000

    # ОПТИМИЗАЦИОННЫЕ ЗАДАЧИ
    rastrigin = RastriginProblem(rastrigin_dimensions)
    rosenbrock = RosenbrockProblem(rosenbrock_dimensions)
    ackley = AckleyProblem(ackley_dimensions)
    schwefel = SchwefelProblem(schwefel_dimensions)
    knapsack = KnapsackProblem(weights,values,capacity)

    problems=[
        rastrigin,
        rosenbrock,
        ackley,
        schwefel,
        knapsack,
    ]

    # ОПТИМИЗАТОРЫ
    optimizers = [
        RandomSearch(iterations=iteration_num),
        SimulatedAnnealing(iterations=iteration_num, T0=1.0, alpha=0.995),
        CrossEntropyMethod(iterations=200, population_size=50, elite_frac=0.2, alpha=0.7, min_sigma=1e-3, min_probability=0.01, max_probability=0.99),
        BasinHopping(iterations=1000, local_steps=20, temperature=1.0),
        AdaptiveRandomSearch(iterations=iteration_num, min_step=1e-6, max_step=10.0, step_increase=1.05, step_decrease=0.95, stagnation_limit=100),
    ]

    benchmark = Benchmark(
        problems=problems,
        optimizers=optimizers,
        runs=200
    )

    results = benchmark.run()
    plot_convergence(results, problems, save_dir="plots", show_std=True)