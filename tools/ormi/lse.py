import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> om.Tensor:
    values = np.asarray(inp["x"], dtype=np.float64)

    if values.ndim != 1 or not values.size:
        raise ValueError("x must be a nonempty vector")

    return om.Tensor(values, dtype=om.float64)


def loss(x: om.Tensor) -> om.Tensor:
    shift = float(x.numpy().max())
    return (x - shift).exp().sum().log() + shift


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def primal(x: om.Tensor) -> om.Tensor:
    return loss(x).realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: result.numpy().tolist())
def gradient(x: om.Tensor) -> om.Tensor:
    return om.grad(loss(x), x).realize()
