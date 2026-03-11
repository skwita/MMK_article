from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, iterations=10000):
        self.iterations = iterations

    @abstractmethod
    def optimize(self, problem):
        pass