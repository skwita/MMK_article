import random
from core.optimizer import Optimizer
from core.result import OptimizationResult
import numpy as np

class MonteCarloMarkovChain(Optimizer):
    def __init__(self, iterations=10000, temperature=1.0):
        super().__init__(iterations)
        self.temperature = temperature

    def optimize(self, problem):
        x = problem.sample_solution()
        fx = problem.evaluate(x)
        best_solution = x
        best_value = fx
        history = []

        for _ in range(self.iterations):
            x_new = problem.neighbor(x)
            fx_new = problem.evaluate(x_new)

            # Метрополис-хастингс
            delta = fx_new - fx
            if ((problem.objective=="min" and (delta < 0 or random.random() < np.exp(-delta / self.temperature))) or
                    (problem.objective=="max" and delta > 0 or random.random() < np.exp(-delta / self.temperature))):
                x = x_new
                fx = fx_new

            if problem.is_better(fx, best_value):
                best_solution = x
                best_value = fx

            history.append(best_value)

        return OptimizationResult(best_solution, best_value, history)