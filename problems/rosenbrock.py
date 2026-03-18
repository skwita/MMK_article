import numpy as np
from core.problem import Problem
from utils import reflect


class RosenbrockProblem(Problem):

    problem_type = "continuous"
    objective = "min"
    initial_step_size = 1

    def __init__(self, dim=10, bounds=(-5, 10)):
        self.dim = dim
        self.bounds = bounds

    def sample_solution(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def evaluate(self, x):
        # f(x) = sum_{i=1}^{n-1} [100*(x_{i+1}-x_i^2)^2 + (x_i - 1)^2]
        return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i]-1)**2 for i in range(self.dim-1))

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