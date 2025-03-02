import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("input", np.float64(input))


def square(server, input):
    (out,), times = futhark_utils.run(
        server, "square", ("output",), ("input",), min_runs=1, min_seconds=0
    )
    return (out, times)


def double(server, input):
    (out,), times = futhark_utils.run(
        server, "double", ("output",), ("input",), min_runs=1, min_seconds=0
    )
    return (out, times)
