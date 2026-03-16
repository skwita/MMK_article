from core.optimizer import Optimizer
from core.result import OptimizationResult
import numpy as np
import random

class MonteCarloTreeSearch(Optimizer):

    def __init__(self, iterations=1000, rollout_depth=5):
        super().__init__(iterations)
        self.rollout_depth = rollout_depth

    def optimize(self, problem):
        best_solution = problem.sample_solution()
        best_value = problem.evaluate(best_solution)
        history = []

        for _ in range(self.iterations):
            x = problem.sample_solution()
            for _ in range(self.rollout_depth):
                x = problem.neighbor(x)
            fx = problem.evaluate(x)

            if problem.is_better(fx, best_value):
                best_solution = x
                best_value = fx

            history.append(best_value)

        return OptimizationResult(best_solution, best_value, history)