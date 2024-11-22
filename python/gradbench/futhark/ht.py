import futhark_server
import numpy as np
from gradbench.adbench.ht_data import HandInput, HandOutput


def prepare(server, params):
    input = HandInput.from_dict(params["input"])

    server.put_value("parents", input.data.model.parents)
    server.put_value("base_relatives", input.data.model.base_relatives)
    server.put_value("inverse_base_absolutes", input.data.model.inverse_base_absolutes)
    server.put_value("weights", input.data.model.weights.T)
    server.put_value("base_positions", input.data.model.base_positions.T)
    server.put_value("triangles", input.data.model.triangles)
    server.put_value("is_mirrored", np.bool_(input.data.model.is_mirrored))
    server.put_value("correspondences", input.data.correspondences)
    server.put_value("points", input.data.points.T)
    server.put_value("theta", input.theta)
    server.put_value("us", input.us.flatten())


def calculate_objectiveHT(server):
    server.cmd_call(
        "calculate_objective",
        "obj",
        "parents",
        "base_relatives",
        "inverse_base_absolutes",
        "weights",
        "base_positions",
        "triangles",
        "is_mirrored",
        "correspondences",
        "points",
        "theta",
        "us",
    )
    obj = server.get_value("obj")
    server.cmd_free("obj")
    return obj.flatten().tolist()


def calculate_jacobianHT(server):
    server.cmd_call(
        "calculate_jacobian",
        "J",
        "parents",
        "base_relatives",
        "inverse_base_absolutes",
        "weights",
        "base_positions",
        "triangles",
        "is_mirrored",
        "correspondences",
        "points",
        "theta",
        "us",
    )
    J = server.get_value("J")
    server.cmd_free("J")
    return J.T.tolist()
