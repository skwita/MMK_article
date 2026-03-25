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



def reflect(x, lower, upper):
    x = np.copy(x)
    width = upper - lower

    if width <= 0:
        raise ValueError("upper must be greater than lower")

    for i in range(len(x)):
        while x[i] < lower or x[i] > upper:
            if x[i] < lower:
                x[i] = lower + (lower - x[i])
            if x[i] > upper:
                x[i] = upper - (x[i] - upper)

    return x