import futhark_utils
from gradbench.adbench.lstm_data import LSTMInput


def prepare(server, input):
    input = LSTMInput.from_dict(input)

    server.put_value("main_params", input.main_params)
    server.put_value("extra_params", input.extra_params)
    server.put_value("state", input.state)
    server.put_value("sequence", input.sequence)


def objective(server, input):
    (obj,), times = futhark_utils.run(
        server,
        "calculate_objective",
        ("obj",),
        ("main_params", "extra_params", "state", "sequence"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (obj, times)


def jacobian(server, input):
    (J,), times = futhark_utils.run(
        server,
        "calculate_jacobian",
        ("J",),
        ("main_params", "extra_params", "state", "sequence"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (J.tolist(), times)
