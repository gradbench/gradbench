import math

import ormi as om

###
###
###


def argmin(
    objective, gradient, start: om.Tensor, *, differentiate: bool = False
) -> om.Tensor:
    x = start
    value = objective(x).item()
    derivative = gradient(x).realize()
    accepted = 0
    rate = 1e-5

    while True:
        if math.sqrt((derivative * derivative).sum().item()) <= 1e-5:
            return x

        if accepted == 10:
            rate *= 2
            accepted = 0
            continue

        # Reuse compiled kernels across updates
        candidate = (x - rate * derivative).realize()
        difference = x - candidate

        if math.sqrt((difference * difference).sum().item()) <= 1e-5:
            return x

        next_value = objective(candidate).item()
        if next_value < value:
            x = candidate

            if not differentiate:
                x = om.Tensor(x.numpy(), dtype=om.float64)

            value = next_value
            derivative = gradient(x).realize()
            accepted += 1
        else:
            rate /= 2
            accepted = 0
