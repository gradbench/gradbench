import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("x", np.array(input["x"], dtype=np.float64))


def primal(server, input):
    (o,), times = futhark_utils.run(
        server,
        "primal",
        ("output",),
        ("x"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (o.tolist(), times)


def gradient(server, input):
    (o,), times = futhark_utils.run(
        server,
        "gradient",
        ("output",),
        ("x"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (o.tolist(), times)
