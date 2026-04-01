import argparse
import random
import time
from typing import Any

import numpy as np

from optimizers.random_search import RandomSearch
from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.ARS import AdaptiveRandomSearch
from optimizers.basin_hopping import BasinHopping
from optimizers.CEM import CrossEntropyMethod

from problems.ackley import AckleyProblem
from problems.rastrigin import RastriginProblem
from problems.rosenbrock import RosenbrockProblem
from problems.schwefel import SchwefelProblem
from problems.knapsack import KnapsackProblem

from utils import generate_knapsack_instance


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_problem(name: str, dim: int, seed: int) -> Any:
    name = name.lower()

    if name == "ackley":
        return AckleyProblem(dim=dim)

    if name == "rastrigin":
        return RastriginProblem(dim=dim)

    if name == "rosenbrock":
        return RosenbrockProblem(dim=dim)

    if name == "schwefel":
        return SchwefelProblem(dim=dim)

    if name == "knapsack":
        weights, values, capacity = generate_knapsack_instance(
            n_items=20,
            capacity_ratio=0.45,
            seed=seed
        )
        return KnapsackProblem(weights, values, capacity)

    raise ValueError(f"Unsupported problem: {name}")


def build_optimizers(problem: Any) -> list[tuple[str, Any]]:
    optimizers = [
        (
            "RandomSearch",
            RandomSearch(iterations=20000)
        ),
        (
            "SimulatedAnnealing",
            SimulatedAnnealing(
                iterations=20000,
                T0=5.0 if problem.problem_type == "continuous" else 1.0,
                alpha=0.995
            )
        ),
        (
            "AdaptiveRandomSearch",
            AdaptiveRandomSearch(
                iterations=20000,
                min_step=1e-6,
                max_step=5.0 if problem.problem_type == "continuous" else 1.0,
                step_increase=1.05,
                step_decrease=0.95,
                stagnation_limit=100
            )
        ),
        (
            "BasinHopping",
            BasinHopping(
                iterations=1000,
                local_steps=20,
                temperature=1.0
            )
        ),
        (
            "CrossEntropyMethod",
            CrossEntropyMethod(
                iterations=200,
                population_size=50,
                elite_frac=0.2,
                alpha=0.7,
                min_sigma=1e-3,
                min_probability=0.01,
                max_probability=0.99
            )
        ),
    ]

    return optimizers


def format_solution(solution: Any, max_len: int = 120) -> str:
    text = np.array2string(np.asarray(solution), precision=6, threshold=20)
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def display_value(problem: Any, value: float) -> float:
    if problem.__class__.__name__ == "KnapsackProblem":
        return -value
    return value


def run_optimizer(problem: Any, optimizer_name: str, optimizer: Any) -> dict[str, Any]:
    start = time.perf_counter()
    result = optimizer.optimize(problem)
    elapsed = time.perf_counter() - start

    return {
        "method": optimizer_name,
        "best_solution": result.best_solution,
        "best_value_raw": result.best_value,
        "best_value_display": display_value(problem, result.best_value),
        "iterations": len(result.history),
        "time_sec": elapsed,
    }


def print_header(problem: Any) -> None:
    print("=" * 100)
    print("Программа для решения задачи стохастической оптимизации на основе метода Монте-Карло")
    print("=" * 100)
    print(f"Задача: {problem.__class__.__name__}")
    if hasattr(problem, "dim"):
        print(f"Размерность: {problem.dim}")
    if problem.__class__.__name__ == "KnapsackProblem":
        print(f"Количество предметов: {problem.n}")
        print(f"Вместимость рюкзака: {problem.capacity}")
    print("-" * 100)


def print_results_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'Метод':28}"
        f"{'Лучшее значение':20}"
        f"{'Итерации':12}"
        f"{'Время, с':12}"
        f"{'Лучшее решение'}"
    )
    print("-" * 100)

    for item in results:
        print(
            f"{item['method']:28}"
            f"{item['best_value_display']:20.6f}"
            f"{item['iterations']:12d}"
            f"{item['time_sec']:12.4f}"
            f"{format_solution(item['best_solution'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запуск стохастических оптимизаторов Монте-Карло на тестовой задаче."
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="ackley",
        choices=["ackley", "rastrigin", "rosenbrock", "schwefel", "knapsack"],
        help="Имя задачи оптимизации."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Размерность непрерывной задачи."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Начальное значение генератора случайных чисел."
    )

    args = parser.parse_args()

    set_seed(args.seed)
    problem = build_problem(args.problem, args.dim, args.seed)
    optimizers = build_optimizers(problem)

    print_header(problem)

    results = []
    for optimizer_name, optimizer in optimizers:
        result = run_optimizer(problem, optimizer_name, optimizer)
        results.append(result)

    if getattr(problem, "objective", "min") == "min":
        results.sort(key=lambda x: x["best_value_raw"])
    else:
        results.sort(key=lambda x: x["best_value_raw"], reverse=True)

    print_results_table(results)
    print("-" * 100)
    print("Выполнение завершено.")


if __name__ == "__main__":
    main()