import numpy as np

from problems.ackley import AckleyProblem
from problems.rastrigin import RastriginProblem
from problems.knapsack import KnapsackProblem

from optimizers.random_search import RandomSearch
from optimizers.simulated_annealing import SimulatedAnnealing

from experiment.benchmark import Benchmark
from problems.rosenbrock import RosenbrockProblem
from problems.schwefel import SchwefelProblem
from utils import generate_knapsack_instance

# ПАРАМЕТРЫ
weights, values, capacity = generate_knapsack_instance(
    n_items=10,
    capacity_ratio=0.45,
    seed=42
)
rastrigin_dimensions  = 2
rosenbrock_dimensions = 2
ackley_dimensions     = 2
schwefel_dimensions   = 2

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
    RandomSearch(10000),
    SimulatedAnnealing(10000),
]

benchmark = Benchmark(
    problems=problems,
    optimizers=optimizers,
    runs=200
)

results = benchmark.run()