import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from experiment.benchmar_results import BenchmarkResult
from config.optimizers_registry import OPTIMIZER_REGISTRY
from config.run_config import EXPERIMENT_CONFIG


def _run_single(problem, optimizer, runs):
    pname = problem.__class__.__name__
    oname = optimizer.__class__.__name__

    scores = []
    times = []
    solutions = []
    histories = []

    for _ in range(runs):

        start = time.time()

        result = optimizer.optimize(problem)

        elapsed = time.time() - start

        scores.append(result.best_value)
        solutions.append(result.best_solution)
        times.append(elapsed)
        histories.append(result.history)

    bench = BenchmarkResult(scores, solutions, histories, objective=problem.objective)

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    return pname, oname, bench, avg_time, total_time

def build_optimizers_for_problem(problem):
    problem_name = problem.__class__.__name__
    problem_config = EXPERIMENT_CONFIG[problem_name]

    optimizers = []
    for optimizer_name, params in problem_config.items():
        optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
        optimizers.append(optimizer_cls(**params))

    return optimizers

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
                optimizers = build_optimizers_for_problem(problem)
                for optimizer in optimizers:

                    futures.append(
                        executor.submit(
                            _run_single,
                            problem,
                            optimizer,
                            self.runs
                        )
                    )

            for future in futures:
                pname, oname, bench, avg_time, total_time = future.result()
                results[pname][oname] = (bench, avg_time, total_time)

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
                f"{'Total time(s)':>15}"
                f"{'Solution':>30}"
            )

            print(header)
            print("-" * len(header))

            for oname in sorted(results[pname].keys()):

                bench, avg_time, total_time = results[pname][oname]

                print(
                    f"{oname:<{method_width}} "
                    f"{bench.mean:>15.4e} "
                    f"{bench.std:>15.4e} "
                    f"{bench.best:>15.4e} "
                    f"{bench.median:>15.4e} "
                    f"{avg_time:>15.3f}"
                    f"{total_time:>15.3f}"
                    f"{bench.best_solution}"
                )