import numpy as np
from core.problem import Problem
from utils import reflect


class AckleyProblem(Problem):

    problem_type = "continuous"
    objective = "min"
    initial_step_size = 1.0

    def __init__(self, dim=10, bounds=(-32.768, 32.768)):
        self.dim = dim
        self.bounds = bounds

    def sample_solution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def evaluate(self, x):
        # f(x) = -20*exp(-0.2*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2*pi*x_i))) + 20 + e
        n = self.dim
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2*np.pi*x))
        return -20 * np.exp(-0.2*np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + 20 + np.e

    def neighbor(self, x, step_size=0.5):
        step = np.random.normal(0, step_size, self.dim)
        # check if inside boundaries and if not reflect
        return reflect(
            x + step,
            self.bounds[0],
            self.bounds[1]
        )

    def is_better(self, a, b):
        return a < b