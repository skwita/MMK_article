import numpy as np
from core.problem import Problem
from utils import reflect


class RastriginProblem(Problem):

    problem_type = "continuous"
    objective = "min"
    initial_step_size = 0.5

    def __init__(self, dim=10, bounds=(-5.12, 5.12)):
        self.dim = dim
        self.bounds = bounds

    def sample_solution(self):
        return np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            self.dim
        )

    def evaluate(self, x):
        A = 10
        return A * self.dim + sum(
            x_i**2 - A * np.cos(2*np.pi*x_i)
            for x_i in x
        )

    def neighbor(self, x, step_size=1):
        step = np.random.uniform(-step_size, step_size, self.dim)
        # check if inside boundaries and if not reflect
        return reflect(
            x + step,
            self.bounds[0],
            self.bounds[1]
        )

    def is_better(self, a, b):
        return a < b