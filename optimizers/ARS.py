import copy
from core.optimizer import Optimizer
from core.result import OptimizationResult


class AdaptiveRandomSearch(Optimizer):

    def __init__(
        self,
        iterations=10000,
        step_increase=1.05,
        step_decrease=0.95,
        stagnation_limit=100
    ):
        super().__init__(iterations)
        self.step_increase = step_increase
        self.step_decrease = step_decrease
        self.stagnation_limit = stagnation_limit

    def optimize(self, problem):

        x = problem.sample_solution()
        value = problem.evaluate(x)

        best = copy.deepcopy(x)
        best_value = value

        history = []
        no_improve = 0

        for _ in range(self.iterations):

            y = problem.neighbor(x)
            y_value = problem.evaluate(y)

            if problem.is_better(y_value, value):
                x = y
                value = y_value
                no_improve = 0

                if problem.is_better(value, best_value):
                    best = copy.deepcopy(x)
                    best_value = value
            else:
                no_improve += 1

            history.append(best_value)

            # Простая адаптация "масштаба" поиска:
            # если долго нет улучшений, делаем рестарт из лучшего/случайного соседа
            if no_improve >= self.stagnation_limit:
                candidate = problem.neighbor(best)
                candidate_value = problem.evaluate(candidate)

                if problem.is_better(candidate_value, value):
                    x = candidate
                    value = candidate_value
                else:
                    x = problem.sample_solution()
                    value = problem.evaluate(x)

                no_improve = 0

        return OptimizationResult(best, best_value, history)