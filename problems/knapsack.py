import numpy as np
from core.problem import Problem


class KnapsackProblem(Problem):

    problem_type = "discrete"
    objective = "max"
    initial_step_size = 1.0

    def __init__(self, weights, values, capacity):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n = len(weights)

    def sample_solution(self):
        return np.random.randint(0, 2, self.n)

    def evaluate(self, x):

        weight = (x * self.weights).sum()
        value = (x * self.values).sum()

        if weight > self.capacity:
            penalty = (weight - self.capacity) * 100
            return value - penalty

        return value

    def neighbor(self, x, step_size=1):
        y = x.copy()

        flips = max(1, int(round(step_size)))
        flips = min(flips, self.n)

        indices = np.random.choice(self.n, size=flips, replace=False)
        y[indices] = 1 - y[indices]

        return y

    def is_better(self, a, b):
        return a > b