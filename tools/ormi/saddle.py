import numpy as np
import ormi as om
from gd import argmin
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> om.Tensor:
    start = np.asarray(inp["start"], dtype=np.float64)

    if start.shape != (2,):
        raise ValueError("start must have shape (2,)")

    return om.Tensor(start, dtype=om.float64)


def objective(x: om.Tensor, y: om.Tensor) -> om.Tensor:
    return (x * x).sum() - (y * y).sum()


def maximize(x: om.Tensor, start: om.Tensor) -> om.Tensor:
    def value(y):
        return -objective(x, y)

    def gradient(y):
        return om.grad(value(y), y)

    # Keep inner updates differentiable
    return argmin(value, gradient, start, differentiate=True)


###
###
###


@wrap.multiple_runs(
    pre=prepare,
    post=lambda result: result[0].numpy().tolist() + result[1].numpy().tolist(),
)
def rr(start: om.Tensor) -> tuple[om.Tensor, om.Tensor]:
    inner_start = om.Tensor(start.numpy().copy(), dtype=om.float64)

    def value(x):
        return objective(x, maximize(x, inner_start))

    def gradient(x):
        return om.grad(value(x), x)

    x = argmin(value, gradient, start)
    y = maximize(x, inner_start)
    return x.realize(), y.realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def ff(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def fr(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def rf(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")
