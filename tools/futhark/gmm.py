import os
import re

import futhark_server
import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("alpha", np.array(input["alpha"], dtype=np.float64))
    server.put_value("means", np.array(input["means"], dtype=np.float64))
    server.put_value("icf", np.array(input["icf"], dtype=np.float64))
    server.put_value("x", np.array(input["x"], dtype=np.float64))
    server.put_value("gamma", np.float64(input["gamma"]))
    server.put_value("m", np.int64(input["m"]))


def calculate_objectiveGMM(server, runs):
    (o,), times = futhark_utils.run(
        server,
        "calculate_objective",
        ("output",),
        ("alpha", "means", "icf", "x", "gamma", "m"),
        runs,
    )
    return (o, times)


def calculate_jacobianGMM(server, runs):
    (o1, o2, o3), times = futhark_utils.run(
        server,
        "calculate_jacobian",
        ("output0", "output1", "output2"),
        ("alpha", "means", "icf", "x", "gamma", "m"),
        runs,
    )
    return (
        o1.flatten().tolist() + o2.flatten().tolist() + o3.flatten().tolist(),
        times,
    )
