import numpy as np


def generate_knapsack_instance(
        n_items=100,
        weight_range=(10, 100),
        value_range=(10, 120),
        capacity_ratio=0.4,
        seed=None
):
    if seed is not None:
        np.random.seed(seed)

    weights = np.random.randint(
        weight_range[0],
        weight_range[1],
        n_items
    )

    values = np.random.randint(
        value_range[0],
        value_range[1],
        n_items
    )

    capacity = int(weights.sum() * capacity_ratio)

    return weights.tolist(), values.tolist(), capacity