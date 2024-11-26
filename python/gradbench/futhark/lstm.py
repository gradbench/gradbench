import futhark_server
import numpy as np

from gradbench.adbench.lstm_data import LSTMInput


def prepare(server, params):
    input = LSTMInput.from_dict(params["input"])

    server.put_value("main_params", input.main_params)
    server.put_value("extra_params", input.extra_params)
    server.put_value("state", input.state)
    server.put_value("sequence", input.sequence)


def calculate_objectiveLSTM(server):
    server.cmd_call(
        "calculate_objective",
        "obj",
        "main_params",
        "extra_params",
        "state",
        "sequence",
    )
    obj = server.get_value("obj")
    server.cmd_free("obj")
    return obj


def calculate_jacobianLSTM(server):
    server.cmd_call(
        "calculate_jacobian",
        "J",
        "main_params",
        "extra_params",
        "state",
        "sequence",
    )
    J = server.get_value("J")
    server.cmd_free("J")
    return J.tolist()
