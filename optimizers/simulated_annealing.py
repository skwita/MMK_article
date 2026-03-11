import math
import random
from core.optimizer import Optimizer
from core.result import OptimizationResult


class SimulatedAnnealing(Optimizer):

    def __init__(self, iterations=10000, T0=1.0, alpha=0.995):
        super().__init__(iterations)
        self.T0 = T0
        self.alpha = alpha

    def optimize(self, problem):

        x = problem.sample_solution()
        value = problem.evaluate(x)

        best = x
        best_value = value

        T = self.T0
        history = []

        for _ in range(self.iterations):

            y = problem.neighbor(x)
            y_value = problem.evaluate(y)

            delta = y_value - value

            if delta < 0 or random.random() < math.exp(-delta / T):
                x = y
                value = y_value

            if problem.is_better(value, best_value):
                best = x
                best_value = value

            T *= self.alpha
            history.append(best_value)

        return OptimizationResult(best, best_value, history)