from optimizers.ARS import AdaptiveRandomSearch
from optimizers.CEM import CrossEntropyMethod
from optimizers.basin_hopping import BasinHopping
from optimizers.random_search import RandomSearch
from optimizers.simulated_annealing import SimulatedAnnealing

OPTIMIZER_REGISTRY = {
    "RandomSearch": RandomSearch,
    "SimulatedAnnealing": SimulatedAnnealing,
    "CrossEntropyMethod": CrossEntropyMethod,
    "BasinHopping": BasinHopping,
    "AdaptiveRandomSearch": AdaptiveRandomSearch,
}