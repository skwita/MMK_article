from abc import ABC
from typing import List
import numpy as np
from model.abstract_func import OptimizationProblem


class RastriginFunc(OptimizationProblem, ABC):
    def calculate(self, x : List[float]) -> float:
        """
        Вычисляет значение функции Растригина для вектора x.
        x: array-like, np.array или list
        """
        x = np.array(x)
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))