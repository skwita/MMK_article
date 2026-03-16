import numpy as np


class BenchmarkResult:

    def __init__(self, scores, solutions, histories, objective = "min"):
        self.scores = np.array(scores)
        self.solutions = np.array(solutions)
        self.objective = objective
        self.histories = [np.array(h, dtype=float) for h in histories]

    @property
    def mean(self):
        return self.scores.mean()

    @property
    def std(self):
        return self.scores.std()

    @property
    def best(self):
        return self.scores.max() if self.objective == "max" else self.scores.min()

    @property
    def best_solution(self):
        idx = np.argmax(self.scores) if self.objective == "max" else np.argmin(self.scores)
        return self.solutions[idx]

    @property
    def median(self):
        return np.median(self.scores)

    def mean_history(self):
        """
        Средняя кривая сходимости по всем запускам.
        Если длины histories разные, обрезаем до минимальной длины.
        """
        if not self.histories:
            return np.array([])

        min_len = min(len(h) for h in self.histories)
        trimmed = np.array([h[:min_len] for h in self.histories], dtype=float)
        return trimmed.mean(axis=0)

    def std_history(self):
        """
        Стандартное отклонение кривой сходимости по всем запускам.
        """
        if not self.histories:
            return np.array([])

        min_len = min(len(h) for h in self.histories)
        trimmed = np.array([h[:min_len] for h in self.histories], dtype=float)
        return trimmed.std(axis=0)

    def __str__(self):
        return (
            f"mean={self.mean:.4f} | "
            f"std={self.std:.4f} | "
            f"best={self.best:.4f} | "
            f"best_solution={self.best_solution} | "
            f"median={self.median:.4f}"
        )