import numpy as np

from core.optimizer import Optimizer
from core.result import OptimizationResult


class CrossEntropyMethod(Optimizer):

    def __init__(
        self,
        iterations=200,
        population_size=50,
        elite_frac=0.2,
        alpha=0.7,
        min_sigma=1e-3,
        min_probability=0.01,
        max_probability=0.99
    ):
        super().__init__(iterations)

        self.population_size = population_size
        self.elite_frac = elite_frac
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.min_probability = min_probability
        self.max_probability = max_probability

    def optimize(self, problem):
        if problem.problem_type == "continuous":
            return self._optimize_continuous(problem)

        if problem.problem_type == "discrete":
            return self._optimize_discrete(problem)

        raise ValueError(
            f"Unsupported problem_type: {problem.problem_type}"
        )

    def _optimize_continuous(self, problem):
        dim = len(problem.sample_solution())
        lower, upper = problem.bounds

        mu = np.random.uniform(lower, upper, dim)
        sigma = np.ones(dim) * (upper - lower) / 2

        best_solution = None
        best_value = float("inf")
        history = []

        elite_size = max(1, int(self.population_size * self.elite_frac))

        for _ in range(self.iterations):
            population = np.random.normal(
                loc=mu,
                scale=sigma,
                size=(self.population_size, dim)
            )

            population = np.clip(population, lower, upper)

            values = np.array([problem.evaluate(x) for x in population])

            elite_idx = np.argsort(values)[:elite_size]
            elite = population[elite_idx]

            new_mu = elite.mean(axis=0)
            new_sigma = elite.std(axis=0)

            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            sigma = self.alpha * sigma + (1 - self.alpha) * new_sigma
            sigma = np.maximum(sigma, self.min_sigma)

            best_idx = np.argmin(values)

            if best_solution is None or problem.is_better(values[best_idx], best_value):
                best_solution = population[best_idx].copy()
                best_value = values[best_idx]

            history.append(best_value)

        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            history=history
        )

    def _optimize_discrete(self, problem):
        dim = len(problem.sample_solution())

        p = np.full(dim, 0.5, dtype=float)

        best_solution = None
        best_value = float("inf")
        history = []

        elite_size = max(1, int(self.population_size * self.elite_frac))

        for _ in range(self.iterations):
            population = np.random.binomial(
                n=1,
                p=p,
                size=(self.population_size, dim)
            )

            values = np.array([problem.evaluate(x) for x in population])

            elite_idx = np.argsort(values)[:elite_size]
            elite = population[elite_idx]

            new_p = elite.mean(axis=0)

            p = self.alpha * p + (1 - self.alpha) * new_p
            p = np.clip(
                p,
                self.min_probability,
                self.max_probability
            )

            best_idx = np.argmin(values)

            if best_solution is None or problem.is_better(values[best_idx], best_value):
                best_solution = population[best_idx].copy()
                best_value = values[best_idx]

            history.append(best_value)

        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            history=history
        )