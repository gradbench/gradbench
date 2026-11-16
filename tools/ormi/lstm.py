import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> dict:
    main = np.asarray(inp["main_params"], dtype=np.float64)
    extra = np.asarray(inp["extra_params"], dtype=np.float64)
    state = np.asarray(inp["state"], dtype=np.float64)
    sequence = np.asarray(inp["sequence"], dtype=np.float64)

    if sequence.ndim != 2 or sequence.shape[0] < 2 or not sequence.shape[1]:
        raise ValueError("sequence needs >= 2 nonempty rows")

    width = sequence.shape[1]

    if (
        main.ndim != 2
        or not main.shape[0]
        or main.shape[0] % 2
        or main.shape[1] != 4 * width
    ):
        raise ValueError("main_params needs row pairs of width 4*b")

    if extra.shape != (3, width) or state.shape != (main.shape[0], width):
        raise ValueError("extra_params and state must match sequence width")

    rows = []

    for row in main:
        rows.append([om.Tensor(part, dtype=om.float64) for part in np.split(row, 4)])

    return {
        "main": rows,
        "extra": [om.Tensor(row, dtype=om.float64) for row in extra],
        "state": [om.Tensor(row, dtype=om.float64) for row in state],
        "sequence": [om.Tensor(row, dtype=om.float64) for row in sequence],
        "width": width,
    }


def loss(data: dict, intermediates: list[om.Tensor] | None = None) -> om.Tensor:
    main = data["main"]
    states = list(data["state"])
    sequence = data["sequence"]
    in_weight, out_weight, out_bias = data["extra"]
    total = om.Tensor(0.0, dtype=om.float64)

    for t in range(len(sequence) - 1):
        value = sequence[t] * in_weight

        if intermediates is not None:
            intermediates.append(value)

        for layer in range(len(main) // 2):
            weight, bias = main[2 * layer], main[2 * layer + 1]
            hidden, cell = states[2 * layer], states[2 * layer + 1]
            forget_input = value * weight[0] + bias[0]
            forget = om.nn.sigmoid(forget_input)
            ingate_input = hidden * weight[1] + bias[1]
            ingate = om.nn.sigmoid(ingate_input)
            outgate_input = value * weight[2] + bias[2]
            outgate = om.nn.sigmoid(outgate_input)
            change_input = hidden * weight[3] + bias[3]
            change = om.nn.tanh(change_input)
            cell = cell * forget + ingate * change
            hidden = outgate * om.nn.tanh(cell)

            # Keep recurrent kernels small
            om.Plan([hidden, cell]).run()

            if intermediates is not None:
                intermediates.extend(
                    (
                        forget_input,
                        ingate_input,
                        outgate_input,
                        change_input,
                        cell,
                        hidden,
                    )
                )

            states[2 * layer], states[2 * layer + 1] = hidden, cell
            value = hidden

        prediction = value * out_weight + out_bias

        if intermediates is not None:
            intermediates.append(prediction)

        # +2 is part of the benchmark
        normalized = prediction - (prediction.exp().sum() + 2).log()
        total = total + (sequence[t + 1] * normalized).sum()

    return -total / ((len(sequence) - 1) * data["width"])


def output_gradients(result: list[om.Tensor]) -> list[float]:
    values = []

    for derivative in result:
        values.extend(derivative.numpy().reshape(-1).tolist())

    return values


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def objective(data: dict) -> om.Tensor:
    return loss(data).realize()


@wrap.multiple_runs(pre=prepare, post=output_gradients)
def jacobian(data: dict) -> list[om.Tensor]:
    inputs = [part for row in data["main"] for part in row] + data["extra"]
    intermediates = []
    value = loss(data, intermediates)
    gradients = om.grad(value, inputs + intermediates)
    derivatives = gradients[: len(inputs)]

    # Run adjoints backward to keep kernels small
    for derivative in reversed(gradients[len(inputs) :]):
        if not derivative.realized:
            derivative.realize()

    for derivative in derivatives:
        derivative.realize()

    return derivatives
