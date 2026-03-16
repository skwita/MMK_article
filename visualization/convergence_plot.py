import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


def _knapsack_optimal_value(problem):
    """
    Точный оптимум для небольшого рюкзака полным перебором.
    У тебя сейчас n_items=10, это 2^10 = 1024 вариантов.
    """
    n = problem.n
    best = -float("inf")

    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        value = problem.evaluate(x)
        if value > best:
            best = value

    return best


def get_optimal_value(problem):
    name = problem.__class__.__name__

    if name in {
        "AckleyProblem",
        "RastriginProblem",
        "RosenbrockProblem",
        "SchwefelProblem",
    }:
        return 0.0

    if name == "KnapsackProblem":
        return float(_knapsack_optimal_value(problem))

    return None


def plot_convergence(results, problems, save_dir="plots", show_std=True):
    """
    Для каждой задачи строит один график:
    - кривые mean(best-so-far) для всех методов
    - горизонтальная линия эталона
    """
    os.makedirs(save_dir, exist_ok=True)

    problems_by_name = {p.__class__.__name__: p for p in problems}

    for problem_name, method_data in results.items():
        problem = problems_by_name[problem_name]
        optimum = get_optimal_value(problem)

        plt.figure(figsize=(10, 6))

        for method_name, (bench, _) in sorted(method_data.items()):
            mean_hist = bench.mean_history()
            std_hist = bench.std_history()

            if len(mean_hist) == 0:
                continue

            x = np.arange(1, len(mean_hist) + 1)

            plt.plot(x, mean_hist, label=method_name)

            if show_std and len(std_hist) == len(mean_hist):
                plt.fill_between(
                    x,
                    mean_hist - std_hist,
                    mean_hist + std_hist,
                    alpha=0.15
                )

        if optimum is not None:
            plt.axhline(
                y=optimum,
                linestyle="--",
                linewidth=2,
                label=f"Optimum = {optimum:.4f}"
            )

        plt.title(f"Convergence on {problem_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far objective value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(save_dir, f"{problem_name}_convergence.png")
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")