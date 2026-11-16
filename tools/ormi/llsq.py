import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def sample_points(n: int) -> np.ndarray:
    if n < 2:
        raise RuntimeError("n must be >= 2")

    idx = np.arange(n, dtype=np.float64)
    return -1 + (idx * 2) / (n - 1)


def basis_matrix(points: np.ndarray, m: int) -> np.ndarray:
    mat = np.empty(shape=(len(points), m), dtype=np.float64)

    for j in range(m):
        mat[:, j] = np.pow(points, j)

    return mat


def loss(x: om.Tensor, basis: om.Tensor, target: om.Tensor) -> om.Tensor:
    sub = target - (basis @ x)
    ss = (sub * sub).sum()
    return ss * 0.5


###
###
###


def prepare(inp: dict) -> tuple[om.Tensor, om.Tensor, om.Tensor]:
    values = np.asarray(inp["x"], dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("x must be a nonempty vector")

    points = sample_points(inp["n"])
    x = om.Tensor(values, dtype=om.float64)
    basis = om.Tensor(basis_matrix(points, len(values)), dtype=om.float64)
    target = om.Tensor(np.sign(points), dtype=om.float64)

    return x, basis, target


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def primal(data: tuple[om.Tensor, om.Tensor, om.Tensor]) -> om.Tensor:
    return loss(*data).realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: result.numpy().tolist())
def gradient(data: tuple[om.Tensor, om.Tensor, om.Tensor]) -> om.Tensor:
    x, basis, target = data
    value = loss(x, basis, target)
    return om.grad(value, x).realize()
