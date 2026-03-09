problems = [RastriginProblem(dim=10), KnapsackProblem(weights, values, capacity)]
optimizers = [MonteCarloDirect, MCMCOptimizer]

runner = ExperimentRunner(problems, optimizers)
runner.run(iterations=1000)

Visualizer.compare_best_values(runner.results)
Visualizer.plot_convergence(runner.results)