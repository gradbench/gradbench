import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> dict:
    theta = np.asarray(inp["theta"], dtype=np.float64)
    if theta.shape != (26,):
        raise ValueError("theta must have shape (26,)")

    model = inp["data"]["model"]
    parents = model["parents"]
    if len(parents) < 21 or any(
        parent >= i or parent < -1 for i, parent in enumerate(parents)
    ):
        raise ValueError("parents must precede children")

    data = {
        "theta": om.Tensor(theta, dtype=om.float64),
        "parents": parents,
        "relative": [
            om.Tensor(value, dtype=om.float64) for value in model["base_relatives"]
        ],
        "inverse": [
            om.Tensor(value, dtype=om.float64)
            for value in model["inverse_base_absolutes"]
        ],
        "positions": np.asarray(model["base_positions"], dtype=np.float64),
        "weights": np.asarray(model["weights"], dtype=np.float64),
        "points": np.asarray(inp["data"]["points"], dtype=np.float64),
        "us": np.asarray(inp["us"], dtype=np.float64),
        "mirrored": model["is_mirrored"],
        "eye3": om.Tensor(np.eye(3), dtype=om.float64),
        "eye4": om.Tensor(np.eye(4), dtype=om.float64),
        "xyz": om.Tensor(np.eye(4)[:, :3].copy(), dtype=om.float64),
        "selectors": [om.Tensor(row, dtype=om.float64) for row in np.eye(26)],
        "u0": om.Tensor([[1.0], [0.0]], dtype=om.float64),
        "u1": om.Tensor([[0.0], [1.0]], dtype=om.float64),
        "b0": om.Tensor([[1.0, 0.0, -1.0]], dtype=om.float64),
        "b1": om.Tensor([[0.0, 1.0, -1.0]], dtype=om.float64),
        "b2": om.Tensor([[0.0, 0.0, 1.0]], dtype=om.float64),
    }
    indices = np.asarray(inp["data"]["correspondences"], dtype=np.intp)

    if data["points"].shape != (len(indices), 3) or not len(indices):
        raise ValueError("Need one 3D point per correspondence")

    if data["us"].size:
        if data["us"].shape != (len(indices), 2):
            raise ValueError("us must have shape (len(correspondences), 2)")

        indices = np.asarray(model["triangles"], dtype=np.intp)[indices].reshape(-1)

    data["indices"] = indices
    data["rotations"] = []

    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        fixed = np.zeros((4, 4), dtype=np.float64)
        fixed[axis, axis] = fixed[3, 3] = 1
        cosine = np.zeros((4, 4), dtype=np.float64)
        cosine[other, other] = 1
        sine = np.zeros((4, 4), dtype=np.float64)
        i, j = other
        sine[i, j], sine[j, i] = (-1, 1) if axis != 1 else (1, -1)
        data["rotations"].append(
            tuple(om.Tensor(value, dtype=om.float64) for value in (fixed, cosine, sine))
        )

    data["skew"] = [
        om.Tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=om.float64
        ),
        om.Tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=om.float64
        ),
        om.Tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=om.float64
        ),
    ]

    return data


def rotation(angle: om.Tensor | None, axis: int, data: dict) -> om.Tensor:
    if angle is None:
        return data["eye4"]

    fixed, cosine, sine = data["rotations"][axis]
    return fixed + cosine * angle.cos() + sine * angle.sin()


def global_rotation(parameters: list[om.Tensor], data: dict) -> om.Tensor:
    norm = (
        parameters[0] * parameters[0]
        + parameters[1] * parameters[1]
        + parameters[2] * parameters[2]
    )

    if norm.item() < 0.0001 * 0.0001:
        return data["eye3"]

    inverse = norm.rsqrt()
    angle = norm * inverse
    axis = [parameter * inverse for parameter in parameters[:3]]
    outer = om.ops.zeros((3, 3), dtype=om.float64)
    skew = om.ops.zeros((3, 3), dtype=om.float64)

    for i in range(3):
        skew = skew + axis[i] * data["skew"][i]

        for j in range(3):
            entry = np.zeros((3, 3), dtype=np.float64)
            entry[i, j] = 1
            outer = outer + axis[i] * axis[j] * om.Tensor(entry, dtype=om.float64)

    cosine, sine = angle.cos(), angle.sin()
    return data["eye3"] * cosine + outer * (1 - cosine) + skew * sine


def residual(data: dict, start: int, stop: int) -> tuple[om.Tensor, om.Tensor | None]:
    parameters = [data["theta"] @ selector for selector in data["selectors"]]
    poses = [[None, None, None] for _ in data["parents"]]
    bone, parameter = 2, 6

    for _ in range(5):
        for joint in range(3):
            poses[bone][0] = parameters[parameter]
            parameter += 1

            if joint == 0:
                poses[bone][1] = parameters[parameter]
                parameter += 1

            bone += 1
        bone += 1

    factor = 3 if data["us"].size else 1
    indices = data["indices"][factor * start : factor * stop]
    base = om.Tensor(data["positions"][indices], dtype=om.float64)
    weights = data["weights"][indices]
    positions = om.ops.zeros((len(indices), 4), dtype=om.float64)
    absolutes = []

    for i, parent in enumerate(data["parents"]):
        x, z, y = poses[i]
        orient = rotation(z, 2, data) @ rotation(y, 1, data) @ rotation(x, 0, data)
        relative = data["relative"][i] @ orient
        absolute = relative if parent == -1 else absolutes[parent] @ relative
        absolutes.append(absolute)
        transform = absolute @ data["inverse"][i]
        weight = om.Tensor(weights[:, i : i + 1].copy(), dtype=om.float64)
        positions = positions + (base @ transform.transpose()) * weight

    positions = positions @ data["xyz"]
    if data["mirrored"]:
        positions = positions * om.Tensor([-1.0, 1.0, 1.0], dtype=om.float64)

    positions = positions @ global_rotation(parameters, data).transpose()
    translation = om.ops.zeros((3,), dtype=om.float64)

    for i in range(3):
        entry = np.zeros(3, dtype=np.float64)
        entry[i] = 1
        translation = translation + parameters[i + 3] * om.Tensor(
            entry, dtype=om.float64
        )

    positions = positions + translation

    us = None

    if data["us"].size:
        us = om.Tensor(data["us"][start:stop], dtype=om.float64)
        barycentric = (
            (us @ data["u0"]) @ data["b0"] + (us @ data["u1"]) @ data["b1"] + data["b2"]
        )
        vertices = positions.reshape((stop - start, 3, 3))
        positions = (barycentric.reshape((stop - start, 1, 3)) @ vertices).reshape(
            (stop - start, 3)
        )

    target = om.Tensor(data["points"][start:stop], dtype=om.float64)
    return target - positions, us


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.reshape(-1).tolist())
def objective(data: dict) -> np.ndarray:
    result = []

    for start in range(0, len(data["points"]), 2048):
        stop = min(start + 2048, len(data["points"]))
        value, _ = residual(data, start, stop)
        result.append(value.numpy())

    return np.concatenate(result, axis=0)


@wrap.multiple_runs(pre=prepare, post=lambda result: result.tolist())
def jacobian(data: dict) -> np.ndarray:
    blocks = []

    for start in range(0, len(data["points"]), 2048):
        stop = min(start + 2048, len(data["points"]))
        value, us = residual(data, start, stop)
        seed = om.Tensor(np.ones(value.shape), dtype=om.float64)
        reverse = om.grad(value, data["theta"], seed)
        # Jacobian columns via seed gradients
        columns = [om.grad(reverse, seed, selector) for selector in data["selectors"]]
        local = []

        if us is not None:
            # Coordinates affect only their own point
            for axis in range(3):
                direction = np.zeros(value.shape, dtype=np.float64)
                direction[:, axis] = 1
                local.append(om.grad(value, us, om.Tensor(direction, dtype=om.float64)))

        om.Plan(columns + local).run()

        matrix = np.stack([column.numpy().reshape(-1) for column in columns], axis=1)

        if local:
            coordinates = np.stack([item.numpy() for item in local], axis=1).reshape(
                (-1, 2)
            )
            matrix = np.concatenate((coordinates, matrix), axis=1)

        blocks.append(matrix)

    return np.concatenate(blocks, axis=0)
