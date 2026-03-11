from abc import ABC, abstractmethod


class Problem(ABC):

    @abstractmethod
    def sample_solution(self):
        """Случайная генерация допустимого решения"""
        pass

    @abstractmethod
    def evaluate(self, solution):
        """Значение целевой функции"""
        pass

    @abstractmethod
    def neighbor(self, solution):
        """Соседнее решение (для MCMC, SA)"""
        pass

    @abstractmethod
    def is_better(self, a, b):
        """Сравнение решений"""
        pass