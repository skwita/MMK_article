EXPERIMENT_CONFIG = {
    "AckleyProblem": {
        "RandomSearch": {
            "iterations": 10000
        },
        "SimulatedAnnealing": {
            "iterations": 40000,
            "T0": 1.0,
            "alpha": 0.995
        },
        "CrossEntropyMethod": {
            "iterations": 200,
            "population_size": 50,
            "elite_frac": 0.2,
            "alpha": 0.7,
            "min_sigma" : 1e-3,
            "min_probability" : 0.01,
            "max_probability" : 0.99
        },
        "BasinHopping": {
            "iterations": 1000,
            "local_steps": 20,
            "temperature": 1.0
        },
        "AdaptiveRandomSearch": {
            "iterations": 40000,
            "min_step": 1e-6,
            "max_step": 10.0,
            "step_increase": 1.05,
            "step_decrease": 0.95,
            "stagnation_limit": 100
        },
    },
    "RastriginProblem": {
        "RandomSearch": {
            "iterations": 40000
        },
        "SimulatedAnnealing": {
            "iterations": 40000,
            "T0": 1.0,
            "alpha": 0.995
        },
        "CrossEntropyMethod": {
            "iterations": 200,
            "population_size": 50,
            "elite_frac": 0.2,
            "alpha": 0.7,
            "min_sigma" : 1e-3,
            "min_probability" : 0.01,
            "max_probability" : 0.99
        },
        "BasinHopping": {
            "iterations": 1000,
            "local_steps": 20,
            "temperature": 1.0
        },
        "AdaptiveRandomSearch": {
            "iterations": 40000,
            "min_step": 1e-6,
            "max_step": 10.0,
            "step_increase": 1.05,
            "step_decrease": 0.95,
            "stagnation_limit": 100
        },
    },
    "RosenbrockProblem": {
        "RandomSearch": {
            "iterations": 40000
        },
        "SimulatedAnnealing": {
            "iterations": 40000,
            "T0": 1.0,
            "alpha": 0.995
        },
        "CrossEntropyMethod": {
            "iterations": 200,
            "population_size": 50,
            "elite_frac": 0.2,
            "alpha": 0.7,
            "min_sigma" : 1e-3,
            "min_probability" : 0.01,
            "max_probability" : 0.99
        },
        "BasinHopping": {
            "iterations": 1000,
            "local_steps": 20,
            "temperature": 1.0
        },
        "AdaptiveRandomSearch": {
            "iterations": 40000,
            "min_step": 1e-6,
            "max_step": 10.0,
            "step_increase": 1.05,
            "step_decrease": 0.95,
            "stagnation_limit": 100
        },
    },
    "SchwefelProblem": {
        "RandomSearch": {
            "iterations": 40000
        },
        "SimulatedAnnealing": {
            "iterations": 40000,
            "T0": 1.0,
            "alpha": 0.995
        },
        "CrossEntropyMethod": {
            "iterations": 200,
            "population_size": 50,
            "elite_frac": 0.2,
            "alpha": 0.7,
            "min_sigma" : 1e-3,
            "min_probability" : 0.01,
            "max_probability" : 0.99
        },
        "BasinHopping": {
            "iterations": 1000,
            "local_steps": 20,
            "temperature": 1.0
        },
        "AdaptiveRandomSearch": {
            "iterations": 40000,
            "min_step": 1e-6,
            "max_step": 10.0,
            "step_increase": 1.05,
            "step_decrease": 0.95,
            "stagnation_limit": 100
        },
    },
    "KnapsackProblem": {
        "RandomSearch": {
            "iterations": 40000
        },
        "SimulatedAnnealing": {
            "iterations": 40000,
            "T0": 1.0,
            "alpha": 0.995
        },
        "CrossEntropyMethod": {
            "iterations": 200,
            "population_size": 50,
            "elite_frac": 0.2,
            "alpha": 0.7,
            "min_sigma" : 1e-3,
            "min_probability" : 0.01,
            "max_probability" : 0.99
        },
        "BasinHopping": {
            "iterations": 1000,
            "local_steps": 20,
            "temperature": 1.0
        },
        "AdaptiveRandomSearch": {
            "iterations": 40000,
            "min_step": 1e-6,
            "max_step": 10.0,
            "step_increase": 1.05,
            "step_decrease": 0.95,
            "stagnation_limit": 100
        },
    },
}