import ormi as om
from gradbench import wrap

###
###
###


@wrap.function(pre=lambda value: value, post=lambda result: result)
def square(inp: float) -> float:
    x = om.Tensor(inp, dtype=om.float64)
    y = x * x
    return y.item()


@wrap.function(pre=lambda value: value, post=lambda result: result)
def double(inp: float) -> float:
    x = om.Tensor(inp, dtype=om.float64)
    y = x * x
    dydx = om.grad(y, x)
    return dydx.item()
