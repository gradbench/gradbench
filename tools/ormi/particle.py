import ormi as om
from gd import argmin
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> om.Tensor:
    return om.Tensor(float(inp["w"]), dtype=om.float64)


def potential(x: om.Tensor, y: om.Tensor, w: om.Tensor) -> om.Tensor:
    dx = x - 10
    dy = y - (10 - w)
    return (dx * dx + y * y).rsqrt() + (dx * dx + dy * dy).rsqrt()


def intersection(w: om.Tensor) -> om.Tensor:
    x = om.Tensor(0.0, dtype=om.float64)
    y = om.Tensor(8.0, dtype=om.float64)
    vx = om.Tensor(0.75, dtype=om.float64)
    vy = om.Tensor(0.0, dtype=om.float64)

    for _ in range(1_000_000):
        dx, dy = om.grad(potential(x, y, w), [x, y])
        next_x, next_y = x + 0.1 * vx, y + 0.1 * vy
        next_vx, next_vy = vx - 0.1 * dx, vy - 0.1 * dy
        # Keep Euler kernels small

        om.Plan([next_x, next_y, next_vx, next_vy]).run()

        if next_y.item() < 0:
            break

        x, y, vx, vy = next_x, next_y, next_vx, next_vy

    final_step = -y / vy
    final_x = x + final_step * vx
    return final_x * final_x


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def rr(start: om.Tensor) -> om.Tensor:
    def gradient(w):
        return om.grad(intersection(w), w)

    return argmin(intersection, gradient, start).realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def ff(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def fr(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def rf(start: om.Tensor):
    raise NotImplementedError("Forward mode is unsupported")
