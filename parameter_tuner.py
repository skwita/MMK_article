import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from optimizers.ARS import AdaptiveRandomSearch
from optimizers.CEM import CrossEntropyMethod
from optimizers.basin_hopping import BasinHopping
from optimizers.simulated_annealing import SimulatedAnnealing

from problems.ackley import AckleyProblem
from problems.knapsack import KnapsackProblem
from problems.rastrigin import RastriginProblem
from problems.rosenbrock import RosenbrockProblem
from problems.schwefel import SchwefelProblem

from utils import generate_knapsack_instance


def sample_param_dicts(param_grid, n_samples, seed=None):
    rng = random.Random(seed)
    keys = list(param_grid.keys())

    samples = []
    for _ in range(n_samples):
        params = {k: rng.choice(param_grid[k]) for k in keys}
        samples.append(params)

    return samples


def evaluate_param_set(args):
    """
    Отдельная функция верхнего уровня нужна для multiprocessing на Windows.
    """
    optimizer_cls, params, problem_factory, runs, seed = args

    scores = []

    for _ in range(runs):
        problem = problem_factory()
        optimizer = optimizer_cls(**params)

        result = optimizer.optimize(problem)
        scores.append(result.best_value)

    score = float(np.median(scores))
    return params, score, scores


class ParameterTuner:
    def __init__(
        self,
        optimizer_cls,
        param_grid,
        runs=20,
        n_samples=50,
        n_jobs=None,
        random_seed=42,
    ):
        self.optimizer_cls = optimizer_cls
        self.param_grid = param_grid
        self.runs = runs
        self.n_samples = n_samples
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self.random_seed = random_seed

    def tune(self, problem_factory):
        """
        problem_factory: функция без аргументов, создающая новый экземпляр задачи.
        """
        problem_probe = problem_factory()

        sampled_params = sample_param_dicts(
            self.param_grid,
            n_samples=self.n_samples,
            seed=self.random_seed
        )

        tasks = []
        for i, params in enumerate(sampled_params):
            tasks.append((
                self.optimizer_cls,
                params,
                problem_factory,
                self.runs,
                self.random_seed + i
            ))

        best_params = None
        best_score = float("inf") if problem_probe.objective == "min" else -float("inf")
        all_results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(evaluate_param_set, task) for task in tasks]

            for future in as_completed(futures):
                params, score, scores = future.result()
                all_results.append({
                    "params": params,
                    "median_score": score,
                    "scores": scores,
                })

                if best_params is None or problem_probe.is_better(score, best_score):
                    best_params = params
                    best_score = score

        all_results.sort(
            key=lambda x: x["median_score"],
            reverse=(problem_probe.objective == "max")
        )

        return best_params, best_score, all_results

def make_ackley():
    return AckleyProblem(dim=2)

def make_rastrigin():
    return RastriginProblem(dim=2)

def make_rosenbrock():
    return RosenbrockProblem(dim=2)

def make_schwefel():
    return SchwefelProblem(dim=2)

def make_knapsack():
    weights, values, capacity = generate_knapsack_instance(
        n_items=20,
        capacity_ratio=0.45,
        seed=42
    )
    return KnapsackProblem(weights, values, capacity)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    grid = {
        SimulatedAnnealing: {
            "iterations": [5000, 10000, 20000],
            "T0": [1.0, 5.0, 10.0, 50.0],
            "alpha": [0.98, 0.99, 0.995, 0.999, 0.9995],
        },
        AdaptiveRandomSearch: {
            "iterations": [20000, 40000, 80000],
            "min_step": [1e-4, 1e-6, 1e-8],
            "max_step": [0.5, 1.0, 5.0, 10.0],
            "step_increase": [1.01, 1.02, 1.05, 1.1],
            "step_decrease": [0.9, 0.95, 0.98],
            "stagnation_limit": [20, 50, 100, 200],
        },
        BasinHopping: {
            "iterations": [500, 1000, 2000, 5000],
            "local_steps": [5, 10, 20, 30, 50],
            "temperature": [0.01, 0.1, 0.5, 1.0, 5.0],
        },
        CrossEntropyMethod: {
            "iterations": [100, 200, 400, 800],
            "population_size": [20, 30, 50, 100, 200],
            "elite_frac": [0.05, 0.1, 0.2, 0.3],
            "alpha": [0.5, 0.6, 0.7, 0.8, 0.9],
            "min_sigma": [1e-1, 1e-2, 1e-3],
            "min_probability": [0.01, 0.02, 0.05],
            "max_probability": [0.95, 0.99],
        },
    }

    problem_factories = {
        "AckleyProblem": make_ackley,
        "RastriginProblem": make_rastrigin,
        "RosenbrockProblem": make_rosenbrock,
        "SchwefelProblem": make_schwefel,
        "KnapsackProblem": make_knapsack,
    }

    optimizers = [
        AdaptiveRandomSearch,
        SimulatedAnnealing,
        CrossEntropyMethod,
        BasinHopping,
    ]

    for optimizer_cls in optimizers:
        print(f"\n=== {optimizer_cls.__name__} ===")

        for problem_name, problem_factory in problem_factories.items():
            print(f"\n--- {problem_name} ---")

            tuner = ParameterTuner(
                optimizer_cls=optimizer_cls,
                param_grid=grid[optimizer_cls],
                runs=100,          # число прогонов на один набор параметров
                n_samples=100,     # сколько случайных наборов параметров пробовать
                n_jobs=None,      # None -> cpu_count()-1
                random_seed=42,
            )

            best_params, best_score, all_results = tuner.tune(problem_factory)

            print("Best params:", best_params)
            print("Best median score:", best_score)

            # Топ-3 конфигурации
            print("Top 3:")
            for item in all_results[:3]:
                print(item["params"], "->", item["median_score"])