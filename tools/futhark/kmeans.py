import os
import re

import futhark_server
import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("k", np.int64(input["k"]))
    server.put_value("points", np.array(input["points"], dtype=np.float32))


def kmeans(server, input):
    runs = input["runs"]
    (o,), times = futhark_utils.run(
        server,
        "kmeans",
        ("output",),
        ("k", "points"),
        runs,
    )
    return (o.tolist(), times)
