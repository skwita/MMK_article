from abc import ABC, abstractmethod

from core.result import OptimizationResult


class Optimizer(ABC):

    def __init__(self, iterations=10000):
        self.iterations = iterations

    @abstractmethod
    def optimize(self, problem) -> OptimizationResult:
        pass