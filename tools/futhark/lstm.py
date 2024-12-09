import futhark_server
import futhark_utils
import numpy as np

from gradbench.adbench.lstm_data import LSTMInput


def prepare(server, input):
    input = LSTMInput.from_dict(input)

    server.put_value("main_params", input.main_params)
    server.put_value("extra_params", input.extra_params)
    server.put_value("state", input.state)
    server.put_value("sequence", input.sequence)


def calculate_objectiveLSTM(server, runs):
    (obj,), times = futhark_utils.run(
        server,
        "calculate_objective",
        ("obj",),
        ("main_params", "extra_params", "state", "sequence"),
        runs,
    )
    return (obj, times)


def calculate_jacobianLSTM(server, runs):
    (J,), times = futhark_utils.run(
        server,
        "calculate_jacobian",
        ("J",),
        ("main_params", "extra_params", "state", "sequence"),
        runs,
    )
    return (J.tolist(), times)
