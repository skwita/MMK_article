import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from experiment.benchmar_results import BenchmarkResult


def _run_single(problem, optimizer, runs):
    pname = problem.__class__.__name__
    oname = optimizer.__class__.__name__

    scores = []
    times = []
    solutions = []

    for _ in range(runs):

        start = time.time()

        result = optimizer.optimize(problem)

        elapsed = time.time() - start

        scores.append(result.best_value)
        solutions.append(result.best_solution)
        times.append(elapsed)

    bench = BenchmarkResult(scores, solutions)

    avg_time = sum(times) / len(times)

    return pname, oname, bench, avg_time


class Benchmark:

    def __init__(self, problems, optimizers, runs=10, workers=None):
        self.problems = problems
        self.optimizers = optimizers
        self.runs = runs
        self.workers = workers

    def run(self):

        results = defaultdict(dict)
        futures = []

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            for problem in self.problems:
                for optimizer in self.optimizers:

                    futures.append(
                        executor.submit(
                            _run_single,
                            problem,
                            optimizer,
                            self.runs
                        )
                    )

            for future in futures:
                pname, oname, bench, avg_time = future.result()
                results[pname][oname] = (bench, avg_time)

        self._print_table(results)

        return results

    def _print_table(self, results):

        method_width = 22

        for pname in sorted(results.keys()):

            print(f"\n=== {pname} ===")

            header = (
                f"{'Method':<{method_width}} "
                f"{'Mean':>15} "
                f"{'Std':>15} "
                f"{'Best':>15} "
                f"{'Median':>15} "
                f"{'Time(s)':>15}"
                f"{'Solution':>15}"
            )

            print(header)
            print("-" * len(header))

            for oname in sorted(results[pname].keys()):

                bench, avg_time = results[pname][oname]

                print(
                    f"{oname:<{method_width}} "
                    f"{bench.mean:>15.4f} "
                    f"{bench.std:>15.4f} "
                    f"{bench.best:>15.4f} "
                    f"{bench.median:>15.4f} "
                    f"{avg_time:>15.3f}"
                    # f"{bench.best_solution}"
                )