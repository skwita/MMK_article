import numpy as np
from core.problem import Problem

class SchwefelProblem(Problem):
    def __init__(self, dim=10, bounds=(-500, 500)):
        self.dim = dim
        self.bounds = bounds

    def sample_solution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def evaluate(self, x):
        # f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
        n = self.dim
        return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def neighbor(self, x):
        step = np.random.normal(0, 5, self.dim)
        return x + step

    def is_better(self, a, b):
        return a < b