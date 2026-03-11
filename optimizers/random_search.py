from core.optimizer import Optimizer
from core.result import OptimizationResult


class RandomSearch(Optimizer):

    def optimize(self, problem):

        best_solution = None
        best_value = float("inf")
        history = []

        for _ in range(self.iterations):

            s = problem.sample_solution()
            value = problem.evaluate(s)

            if best_solution is None or problem.is_better(value, best_value):
                best_solution = s
                best_value = value

            history.append(best_value)

        return OptimizationResult(
            best_solution,
            best_value,
            history
        )