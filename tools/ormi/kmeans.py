import numpy as np
import ormi as om
from gradbench import wrap

###
###
###


def prepare(inp: dict) -> tuple[np.ndarray, om.Tensor]:
    points = np.asarray(inp["points"], dtype=np.float64)
    centers = np.asarray(inp["centroids"], dtype=np.float64)

    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("points and centroids must be matrices")

    if not points.size or not centers.size or points.shape[1] != centers.shape[1]:
        raise ValueError("points and centroids must be nonempty with equal width")

    return points, om.Tensor(centers, dtype=om.float64)


def nearest(points: np.ndarray, centroids: om.Tensor) -> tuple[np.ndarray, np.ndarray]:
    n, d = points.shape
    k = centroids.shape[0]
    labels = np.empty(n, dtype=np.intp)
    minimum = np.empty(n, dtype=np.float64)
    centers = centroids.reshape((1, k, d))

    # About 1M distance elements per batch
    rows = max(1, (1 << 20) // (k * d))
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        block = om.Tensor(points[start:stop], dtype=om.float64)
        delta = block.reshape((stop - start, 1, d)) - centers
        distance = (delta * delta).sum([2]).numpy()
        indices = distance.argmin(axis=1)
        labels[start:stop] = indices
        minimum[start:stop] = distance[np.arange(stop - start), indices]

    return labels, minimum


###
###
###


@wrap.multiple_runs(pre=prepare, post=lambda result: result.item())
def cost(data: tuple[np.ndarray, om.Tensor]) -> om.Tensor:
    points, centroids = data
    _, minimum = nearest(points, centroids)
    return om.Tensor(minimum, dtype=om.float64).sum().realize()


@wrap.multiple_runs(pre=prepare, post=lambda result: result.numpy().tolist())
def dir(data: tuple[np.ndarray, om.Tensor]) -> om.Tensor:
    points, centroids = data
    labels, _ = nearest(points, centroids)
    k, d = centroids.shape

    if np.any(np.bincount(labels, minlength=k) == 0):
        raise ValueError("Each centroid needs a nonempty cluster")

    first = om.Tensor(np.zeros((k, d)), dtype=om.float64)
    second = om.Tensor(np.zeros((k, d)), dtype=om.float64)
    rows = max(1, (1 << 20) // (k * d))

    for start in range(0, len(points), rows):
        stop = min(start + rows, len(points))
        assignment = np.zeros((stop - start, k), dtype=np.float64)
        assignment[np.arange(stop - start), labels[start:stop]] = 1

        selected = om.Tensor(assignment, dtype=om.float64) @ centroids
        residual = om.Tensor(points[start:stop], dtype=om.float64) - selected
        value = (residual * residual).sum()
        derivative = om.grad(value, centroids)
        # Fixed assignments give a diagonal Hessian
        diagonal = om.grad(derivative.sum(), centroids)
        first.copy_from((first + derivative).realize())
        second.copy_from((second + diagonal).realize())

    return (first / second).realize()
