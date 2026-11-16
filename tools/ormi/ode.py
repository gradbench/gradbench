import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> tuple[om.Tensor, int]:
    values = np.asarray(inp["x"], dtype=np.float64)

    if values.ndim != 1 or not values.size or inp["s"] < 1:
        raise ValueError("x must be a nonempty vector, s >= 1")

    return om.Tensor(values, dtype=om.float64), inp["s"]


def rhs(x: om.Tensor, y: om.Tensor) -> om.Tensor:
    return om.concat([x[:1], x[1:] * y[:-1]])


def solve(x: om.Tensor, steps: int, *, execute: bool = False) -> om.Tensor:
    step = 2.0 / steps
    y = om.ops.zeros(x.shape, dtype=om.float64)

    for _ in range(steps):
        k1 = rhs(x, y)
        k2 = rhs(x, y + step * k1 / 2)
        k3 = rhs(x, y + step * k2 / 2)
        k4 = rhs(x, y + step * k3)
        y = y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if execute and not y.realized:
            om.Plan([y]).run()

    return y


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.numpy().tolist())
def primal(data: tuple[om.Tensor, int]) -> om.Tensor:
    result = solve(*data, execute=True)

    if not result.realized:
        om.Plan([result]).run()

    return result


@wrap.multiple_runs(pre=prepare, post=lambda result: result.numpy().tolist())
def gradient(data: tuple[om.Tensor, int]) -> om.Tensor:
    x, steps = data
    result = solve(x, steps)
    derivatives = om.grad(result[-1], x)

    if not derivatives.realized:
        om.Plan([derivatives]).run()

    return derivatives
