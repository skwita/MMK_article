from abc import ABC, abstractmethod

from model.abstract_func import OptimizationProblem


class Optimizer(ABC):
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem

    @abstractmethod
    def run(self, iterations: int):
        """Запускает алгоритм оптимизации"""
        pass