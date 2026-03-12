import numpy as np
from core.problem import Problem


class KnapsackProblem(Problem):

    problem_type = "discrete"

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

    def neighbor(self, x):
        y = x.copy()
        i = np.random.randint(self.n)
        y[i] = 1 - y[i]
        return y

    def is_better(self, a, b):
        return a > b