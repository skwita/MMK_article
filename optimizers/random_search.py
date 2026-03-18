from core.optimizer import Optimizer
from core.result import OptimizationResult


class RandomSearch(Optimizer):

    def optimize(self, problem):

        best_solution = problem.sample_solution()
        best_value = problem.evaluate(best_solution)
        history = []

        for _ in range(self.iterations):
            s = problem.sample_solution()
            value = problem.evaluate(s)

            if problem.is_better(value, best_value):
                best_solution = s
                best_value = value

            history.append(best_value)

        return OptimizationResult(
            best_solution,
            best_value,
            history
        )