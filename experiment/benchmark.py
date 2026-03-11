import time
from collections import defaultdict
from experiment.benchmar_results import BenchmarkResult


class Benchmark:

    def __init__(self, problems, optimizers, runs=10):
        self.problems = problems
        self.optimizers = optimizers
        self.runs = runs

    def run(self):

        results = defaultdict(dict)

        for problem in self.problems:

            pname = problem.__class__.__name__

            print(f"\n=== {pname} ===")

            for optimizer in self.optimizers:

                oname = optimizer.__class__.__name__

                scores = []
                times = []
                solutions = []

                for _ in range(self.runs):

                    start = time.time()

                    result = optimizer.optimize(problem)

                    elapsed = time.time() - start

                    scores.append(result.best_value)
                    solutions.append(result.best_solution)
                    times.append(elapsed)

                bench = BenchmarkResult(scores, solutions)

                results[pname][oname] = bench

                print(
                    f"{oname:20} "
                    f"{bench} "
                    f"| time={sum(times)/len(times):.3f}s"
                )

        return results