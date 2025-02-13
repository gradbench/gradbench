import os
import re

import futhark_server
import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("points", np.array(input["points"], dtype=np.float64))
    server.put_value("centroids", np.array(input["centroids"], dtype=np.float64))


def cost(server, input):
    runs = input["runs"]
    (o,), times = futhark_utils.run(
        server,
        "cost",
        ("output",),
        ("points", "centroids"),
        runs,
    )
    return (o.tolist(), times)


def dir(server, input):
    runs = input["runs"]
    (o,), times = futhark_utils.run(
        server,
        "dir",
        ("output",),
        ("points", "centroids"),
        runs,
    )
    return (o.tolist(), times)
