import numpy as np

class OptimizationResult:

    def __init__(self, best_solution, best_value, history):
        self.best_solution = best_solution
        self.best_value = best_value
        self.history = history

    def __str__(self):
        hist = np.array(self.history)

        return (
            "OptimizationResult\n"
            "------------------\n"
            f"best_value      : {self.best_value}\n"
            f"iterations      : {len(self.history)}\n"
            f"start_value     : {hist[0]}\n"
            f"final_value     : {hist[-1]}\n"
            f"best_solution   : {self.best_solution}\n"
        )