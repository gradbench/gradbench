import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> list[list[om.Tensor]]:
    size = inp["ell"]
    values = np.asarray(inp["A"], dtype=np.float64)

    if size < 1 or values.shape != (size * size,):
        raise ValueError("A must contain ell^2 elements")

    matrix = []
    for row in values.reshape((size, size)):
        matrix.append([om.Tensor(float(value), dtype=om.float64) for value in row])

    return matrix


def determinant(
    matrix: list[list[om.Tensor]], intermediates: list[om.Tensor] | None = None
) -> om.Tensor:
    if len(matrix) == 1:
        return matrix[0][0]

    minors = {}

    def expand(columns: tuple[int, ...]) -> om.Tensor:
        if len(columns) == 1:
            return matrix[-1][columns[0]]
        if columns in minors:
            return minors[columns]

        row = len(matrix) - len(columns)
        result = om.Tensor(0.0, dtype=om.float64)

        for offset, column in enumerate(columns):
            remaining = columns[:offset] + columns[offset + 1 :]
            term = matrix[row][column] * expand(remaining)
            result = result - term if offset % 2 else result + term

        minors[columns] = result

        if intermediates is not None:
            intermediates.append(result)

        return result

    return expand(tuple(range(len(matrix))))


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def primal(matrix: list[list[om.Tensor]]) -> om.Tensor:
    intermediates = []
    value = determinant(matrix, intermediates)

    for intermediate in intermediates:
        intermediate.realize()

    return value.realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: [value.item() for value in result])
def gradient(matrix: list[list[om.Tensor]]) -> list[om.Tensor]:
    inputs = [value for row in matrix for value in row]
    intermediates = []
    value = determinant(matrix, intermediates)

    for intermediate in intermediates:
        intermediate.realize()

    gradients = om.grad(value, inputs + intermediates)
    derivatives = gradients[: len(inputs)]

    for derivative in reversed(gradients[len(inputs) :]):
        if not derivative.realized:
            derivative.realize()

    for derivative in derivatives:
        derivative.realize()

    return derivatives
