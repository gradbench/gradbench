import jax
import jax.numpy as jnp
import numpy as np
from gradbench import wrap
from jax import grad, jit, jvp

jax.config.update("jax_enable_x64", True)


def costfun(points, centers):
    def all_pairs_norm(a, b):
        a_sqr = jnp.sum(a**2, 1)[None, :]
        b_sqr = jnp.sum(b**2, 1)[:, None]
        diff = jnp.matmul(a, b.T).T
        return a_sqr + b_sqr - 2 * diff

    dists = all_pairs_norm(points, centers)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


def prepare_input(input):
    centroids = np.array(input["centroids"], dtype=np.float64)
    points = np.array(input["points"], dtype=np.float64)
    return points, centroids


@wrap.multiple_runs(pre=prepare_input, post=lambda x: float(x))
@jit
def cost(input):
    points, centroids = input
    return costfun(points, centroids)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
@jit
def dir(input):
    points, centroids = input
    f_diff = grad(lambda cs: costfun(points, cs))
    d, hes = jvp(f_diff, [centroids], [jnp.ones(shape=centroids.shape)])
    return d / hes
