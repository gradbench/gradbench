import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("ell", np.array(input["ell"], dtype=np.int64))
    server.put_value("A", np.array(input["A"], dtype=np.float64))


def primal(server, input):
    (o,), times = futhark_utils.run(
        server,
        "primal",
        ("output",),
        ("ell", "A"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (o, times)


def gradient(server, input):
    (o,), times = futhark_utils.run(
        server,
        "gradient",
        ("output",),
        ("ell", "A"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (o.flatten().tolist(), times)
