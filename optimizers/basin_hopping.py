import copy
import math
import random
from core.optimizer import Optimizer
from core.result import OptimizationResult


class BasinHopping(Optimizer):

    def __init__(self, iterations=1000, local_steps=20, temperature=1.0):
        super().__init__(iterations)
        self.local_steps = local_steps
        self.temperature = temperature

    def _local_search(self, problem, x):
        current = copy.deepcopy(x)
        current_value = problem.evaluate(current)

        best_local = copy.deepcopy(current)
        best_local_value = current_value

        for _ in range(self.local_steps):
            candidate = problem.neighbor(current)
            candidate_value = problem.evaluate(candidate)

            if problem.is_better(candidate_value, current_value):
                current = candidate
                current_value = candidate_value

                if problem.is_better(current_value, best_local_value):
                    best_local = copy.deepcopy(current)
                    best_local_value = current_value

        return best_local, best_local_value

    def worsening(self, current, candidate, problem):
        if problem.objective == "min":
            return candidate - current
        else:
            return current - candidate

    def optimize(self, problem):

        x = problem.sample_solution()
        x, value = self._local_search(problem, x)

        best = copy.deepcopy(x)
        best_value = value

        history = []

        for _ in range(self.iterations):

            # глобальный "прыжок"
            y = problem.neighbor(x)

            # локальная оптимизация после прыжка
            y, y_value = self._local_search(problem, y)

            accepted = False

            if problem.is_better(y_value, value):
                accepted = True
            else:
                delta = self.worsening(value, y_value, problem)  # delta > 0 для ухудшения
                try:
                    accepted = random.random() < math.exp(-delta / self.temperature)
                except OverflowError:
                    accepted = False

            if accepted:
                x = y
                value = y_value

            if problem.is_better(value, best_value):
                best = copy.deepcopy(x)
                best_value = value

            history.append(best_value)

        return OptimizationResult(best, best_value, history)