import futhark_utils
import numpy as np


def prepare(server, input):
    server.put_value("alpha", np.array(input["alpha"], dtype=np.float64))
    server.put_value("mu", np.array(input["mu"], dtype=np.float64))
    server.put_value("q", np.array(input["q"], dtype=np.float64))
    server.put_value("l", np.array(input["l"], dtype=np.float64))
    server.put_value("x", np.array(input["x"], dtype=np.float64))
    server.put_value("gamma", np.float64(input["gamma"]))
    server.put_value("m", np.int64(input["m"]))


def objective(server, input):
    (o,), times = futhark_utils.run(
        server,
        "calculate_objective",
        ("output",),
        ("alpha", "mu", "q", "l", "x", "gamma", "m"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (o, times)


def jacobian(server, input):
    (alpha_d, mu_d, q_d, l_d), times = futhark_utils.run(
        server,
        "calculate_jacobian",
        ("output0", "output1", "output2", "output3"),
        ("alpha", "mu", "q", "l", "x", "gamma", "m"),
        input["min_runs"],
        input["min_seconds"],
    )
    return (
        {
            "alpha": alpha_d.tolist(),
            "mu": mu_d.tolist(),
            "q": q_d.tolist(),
            "l": l_d.tolist(),
        },
        times,
    )
