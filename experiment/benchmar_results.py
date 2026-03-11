import numpy as np


class BenchmarkResult:

    def __init__(self, scores, solutions):
        self.scores = np.array(scores)
        self.solutions = np.array(solutions)

    @property
    def mean(self):
        return self.scores.mean()

    @property
    def std(self):
        return self.scores.std()

    @property
    def best(self):
        return self.scores.min()

    @property
    def best_solution(self):
        indices = np.where(self.scores == self.best)[0]
        idx = indices[0] if len(indices) > 0 else -1
        return self.solutions[idx]

    @property
    def median(self):
        return np.median(self.scores)

    def __str__(self):
        return (
            f"mean={self.mean:.4f} | "
            f"std={self.std:.4f} | "
            f"best={self.best:.4f} | "
            f"best_solution={self.best_solution} | "
            f"median={self.median:.4f}"
        )