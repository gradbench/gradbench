import math

import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> dict:
    d, k, n = inp["d"], inp["k"], inp["n"]
    if min(d, k, n) < 1 or inp["gamma"] <= 0 or inp["m"] < 0:
        raise ValueError("Expected d, k, n >= 1, gamma > 0, m >= 0")

    data = dict(inp)
    shapes = {
        "x": (n, d),
        "alpha": (k,),
        "mu": (k, d),
        "q": (k, d),
        "l": (k, d * (d - 1) // 2),
    }

    for name, shape in shapes.items():
        values = np.asarray(inp[name], dtype=np.float64)

        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

        data[name] = om.Tensor(values, dtype=om.float64)

    diagonal = np.zeros((d, d * d), dtype=np.float64)
    diagonal[np.arange(d), np.arange(d) * (d + 1)] = 1
    lower = np.zeros((d * (d - 1) // 2, d * d), dtype=np.float64)
    index = 0

    for column in range(d):
        for row in range(column + 1, d):
            lower[index, row * d + column] = 1
            index += 1

    data["diagonal_basis"] = om.Tensor(diagonal, dtype=om.float64)
    data["lower_basis"] = om.Tensor(lower, dtype=om.float64)
    degrees = d + inp["m"] + 1
    log_gamma = d * (d - 1) * math.log(math.pi) / 4

    for j in range(d):
        log_gamma += math.lgamma((degrees - j) / 2)

    data["constant"] = -n * d * math.log(2 * math.pi) / 2
    data["constant"] += k * (
        degrees * d * math.log(inp["gamma"] / math.sqrt(2)) - log_gamma
    )
    return data


def logsumexp(x: om.Tensor, axis: int) -> om.Tensor:
    shift = om.Tensor(x.numpy().max(axis=axis, keepdims=True), dtype=om.float64)
    return (x - shift).exp().sum([axis], keepdims=True).log() + shift


def point_loss(data: dict, points: om.Tensor) -> om.Tensor:
    d, k, n = data["d"], data["k"], points.shape[0]
    alpha, mu, q, lower = (data[name] for name in ("alpha", "mu", "q", "l"))
    diagonal = q.exp()
    matrix = diagonal @ data["diagonal_basis"]

    if d > 1:
        matrix = matrix + lower @ data["lower_basis"]
    matrix = matrix.reshape((1, k, d, d))

    delta = points.reshape((n, 1, d)) - mu.reshape((1, k, d))
    transformed = (matrix @ delta.reshape((n, k, d, 1))).reshape((n, k, d))
    sum_q = q.sum([1])
    score = alpha + sum_q - (transformed * transformed).sum([2]) * 0.5

    return logsumexp(score, 1).sum()


def prior_loss(data: dict) -> om.Tensor:
    diagonal = data["q"].exp()
    lower = data["l"]
    squares = (diagonal * diagonal).sum()

    if data["d"] > 1:
        squares = squares + (lower * lower).sum()

    prior = (
        squares * (0.5 * data["gamma"] * data["gamma"]) - data["q"].sum() * data["m"]
    )

    return data["constant"] - prior - data["n"] * logsumexp(data["alpha"], 0).sum()


def loss(data: dict) -> om.Tensor:
    return point_loss(data, data["x"]) + prior_loss(data)


def accumulate_gradients(
    value: om.Tensor, inputs: list[om.Tensor], totals: list[om.Tensor]
) -> None:
    derivatives = om.grad(value, inputs)
    om.Plan(derivatives).run()

    for total, derivative in zip(totals, derivatives):
        total.copy_from((total + derivative).realize())


def output_gradients(result: dict) -> dict:
    values = {}

    for name, derivative in result.items():
        values[name] = derivative.numpy().tolist()

    return values


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def objective(data: dict) -> om.Tensor:
    total = om.Tensor(0.0, dtype=om.float64)
    total.copy_from(prior_loss(data).realize())
    points = data["x"].numpy()
    # Cap batch memory
    rows = max(1, (1 << 24) // (data["k"] * data["d"] * data["d"]))

    for start in range(0, len(points), rows):
        block = om.Tensor(points[start : start + rows], dtype=om.float64)
        total.copy_from((total + point_loss(data, block)).realize())

    return total


@wrap.multiple_runs(pre=prepare, post=output_gradients)
def jacobian(data: dict) -> dict:
    names = ("alpha", "mu", "q", "l")
    inputs = [data[name] for name in names]
    totals = [om.Tensor(np.zeros(value.shape), dtype=om.float64) for value in inputs]

    accumulate_gradients(prior_loss(data), inputs, totals)

    points = data["x"].numpy()
    rows = max(1, (1 << 24) // (data["k"] * data["d"] * data["d"]))

    for start in range(0, len(points), rows):
        block = om.Tensor(points[start : start + rows], dtype=om.float64)
        accumulate_gradients(point_loss(data, block), inputs, totals)

    return dict(zip(names, totals))
