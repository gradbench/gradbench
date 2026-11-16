import numpy as np
import ormi as om
from gradbench import wrap
from gradbench.adbench.ba_sparse_mat import BASparseMat

###
###
###


def prepare(inp: dict) -> dict:
    if min(inp["n"], inp["m"], inp["p"]) < 1:
        raise ValueError("n, m and p must be positive")

    data = dict(inp)
    eye = np.eye(11, dtype=np.float64)

    for name, columns in (
        ("rotation", [0, 1, 2]),
        ("center", [3, 4, 5]),
        ("focal", [6]),
        ("principal", [7, 8]),
        ("radial", [9, 10]),
    ):
        data[name] = om.Tensor(eye[columns].T.copy(), dtype=om.float64)

    eye = np.eye(3, dtype=np.float64)
    data["cycle1"] = om.Tensor(eye[:, [1, 2, 0]], dtype=om.float64)
    data["cycle2"] = om.Tensor(eye[:, [2, 0, 1]], dtype=om.float64)
    data["xy"] = om.Tensor(eye[:, :2].copy(), dtype=om.float64)
    data["z"] = om.Tensor(eye[:, 2:].copy(), dtype=om.float64)
    data["r0"] = om.Tensor([[1.0], [0.0]], dtype=om.float64)
    data["r1"] = om.Tensor([[0.0], [1.0]], dtype=om.float64)
    return data


def cross(a: om.Tensor, b: om.Tensor, data: dict) -> om.Tensor:
    return (a @ data["cycle1"]) * (b @ data["cycle2"]) - (a @ data["cycle2"]) * (
        b @ data["cycle1"]
    )


def inputs(data: dict, count: int) -> tuple[om.Tensor, om.Tensor, om.Tensor, om.Tensor]:
    camera = om.Tensor(np.tile(data["cam"], (count, 1)), dtype=om.float64)
    point = om.Tensor(np.tile(data["x"], (count, 1)), dtype=om.float64)
    weight = om.Tensor(np.full((count, 1), data["w"]), dtype=om.float64)
    feature = om.Tensor(np.tile(data["feat"], (count, 1)), dtype=om.float64)
    return camera, point, weight, feature


def errors(
    camera: om.Tensor,
    point: om.Tensor,
    weight: om.Tensor,
    feature: om.Tensor,
    data: dict,
) -> tuple[om.Tensor, om.Tensor]:
    rotation = camera @ data["rotation"]
    translated = point - camera @ data["center"]
    norm = (rotation * rotation).sum([1], keepdims=True)
    active = (norm.numpy() != 0).astype(np.float64)
    mask = om.Tensor(active, dtype=om.float64)
    safe = norm + om.Tensor(1 - active, dtype=om.float64)
    inverse = safe.rsqrt()
    angle = safe * inverse
    cosine, sine = angle.cos(), angle.sin()
    product = cross(rotation, translated, data)
    full = translated * cosine + product * (sine * inverse)
    full = full + rotation * (
        (rotation * translated).sum([1], keepdims=True) * (1 - cosine) / safe
    )
    rotated = full * mask + (translated + product) * (1 - mask)

    projected = (rotated @ data["xy"]) / (rotated @ data["z"])
    radius = (projected * projected).sum([1], keepdims=True)
    radial = camera @ data["radial"]
    distortion = (
        1 + (radial @ data["r0"]) * radius + (radial @ data["r1"]) * radius * radius
    )
    prediction = (
        projected * distortion * (camera @ data["focal"]) + camera @ data["principal"]
    )
    return weight * (prediction - feature), 1 - weight * weight


def compressed(matrix: BASparseMat) -> dict:
    result = {}

    for name, convert in (("rows", int), ("cols", int), ("vals", float)):
        vector = getattr(matrix, name)
        result[name] = [convert(vector[i]) for i in range(min(30, len(vector)))]
        result[name].append(convert(vector.get_last()))

    return result


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result)
def objective(data: dict) -> dict:
    first = None

    for start in range(0, data["p"], 16384):
        count = min(16384, data["p"] - start)
        reprojection, weight = errors(*inputs(data, count), data)

        om.Plan([reprojection, weight]).run()

        if first is None:
            first = (reprojection.numpy()[0].tolist(), float(weight.numpy()[0, 0]))

    return {
        "reproj_error": {"elements": first[0], "repeated": data["p"]},
        "w_err": {"element": first[1], "repeated": data["p"]},
    }


@wrap.multiple_runs(pre=prepare, post=compressed)
def jacobian(data: dict) -> BASparseMat:
    matrix = BASparseMat(data["n"], data["m"], data["p"])
    weights = []

    for start in range(0, data["p"], 16384):
        count = min(16384, data["p"] - start)
        camera, point, weight, feature = inputs(data, count)
        reprojection, weight_error = errors(camera, point, weight, feature, data)
        columns = []

        for axis in range(2):
            seed = np.zeros((count, 2), dtype=np.float64)
            seed[:, axis] = 1
            columns.append(
                om.grad(
                    output=reprojection,
                    inputs=[camera, point, weight],
                    seed=om.Tensor(seed, dtype=om.float64),
                )
            )

        weight_gradient = om.grad(weight_error.sum(), weight)

        om.Plan(columns[0] + columns[1] + [weight_gradient]).run()

        first = np.concatenate([item.numpy() for item in columns[0]], axis=1)
        second = np.concatenate([item.numpy() for item in columns[1]], axis=1)
        blocks = np.stack((first, second), axis=2)

        for row in range(count):
            index = start + row

            matrix.insert_reproj_err_block(
                index, index % data["n"], index % data["m"], blocks[row].reshape(-1)
            )

        weights.extend(weight_gradient.numpy().reshape(-1).tolist())

    for index, derivative in enumerate(weights):
        matrix.insert_w_err_block(index, derivative)
    return matrix
