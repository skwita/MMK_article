from abc import ABC, abstractmethod


class OptimizationProblem(ABC):
    @abstractmethod
    def evaluate(self, x):
        """Возвращает значение целевой функции в точке x"""
        pass

    @abstractmethod
    def get_bounds(self):
        """Возвращает границы переменных (для непрерывных задач)"""
        pass