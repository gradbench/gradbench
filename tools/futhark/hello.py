import futhark_server
import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("input", np.float64(input))


def square(server, input):
    runs = 1
    (out,), times = futhark_utils.run(
        server,
        "square",
        ("output",),
        ("input",),
        runs,
    )
    return (out, times)


def double(server, input):
    runs = 1
    (out,), times = futhark_utils.run(server, "double", ("output",), ("input",), runs)
    return (out, times)
