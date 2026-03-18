import copy

from core.optimizer import Optimizer
from core.result import OptimizationResult


class AdaptiveRandomSearch(Optimizer):

    def __init__(
        self,
        iterations=10000,
        min_step=1e-6,
        max_step=10.0,
        step_increase=1.05,
        step_decrease=0.95,
        stagnation_limit=100
    ):
        super().__init__(iterations)

        self.min_step = min_step
        self.max_step = max_step
        self.step_increase = step_increase
        self.step_decrease = step_decrease
        self.stagnation_limit = stagnation_limit

    def optimize(self, problem):

        x = problem.sample_solution()
        value = problem.evaluate(x)

        best = copy.deepcopy(x)
        best_value = value

        history = []
        no_improve = 0
        step_size = problem.initial_step_size

        for _ in range(self.iterations):

            y = problem.neighbor(x, step_size)
            y_value = problem.evaluate(y)

            if problem.is_better(y_value, value):
                x = y
                value = y_value
                no_improve = 0

                step_size = min(step_size * self.step_increase, self.max_step)

                if problem.is_better(value, best_value):
                    best = copy.deepcopy(x)
                    best_value = value
            else:
                no_improve += 1
                step_size = max(step_size * self.step_decrease, self.min_step)

            history.append(best_value)

            if no_improve >= self.stagnation_limit:
                candidate = problem.neighbor(best, step_size)
                candidate_value = problem.evaluate(candidate)

                if problem.is_better(candidate_value, value):
                    x = candidate
                    value = candidate_value
                else:
                    x = problem.sample_solution()
                    value = problem.evaluate(x)

                step_size = problem.initial_step_size
                no_improve = 0

        return OptimizationResult(best, best_value, history)