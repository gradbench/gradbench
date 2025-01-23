import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, jvp

from gradbench import wrap


def cost(points, centers):
    def all_pairs_norm(a, b):
        a_sqr = jnp.sum(a**2, 1)[None, :]
        b_sqr = jnp.sum(b**2, 1)[:, None]
        diff = jnp.matmul(a, b.T).T
        return a_sqr + b_sqr - 2 * diff

    dists = all_pairs_norm(points, centers)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


def prepare_input(input):
    k = input["k"]
    features = np.array(input["points"], dtype=np.float64)
    clusters = np.flip(features[-int(k) :], (0,))
    return k, clusters, features


@wrap.multiple_runs(
    runs=lambda x: x["runs"], pre=prepare_input, post=lambda x: x.tolist()
)
@jit
def kmeans(input):
    k, clusters, features = input
    max_iter = 10
    tolerance = 1.0

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        f_diff = grad(lambda cs: cost(features, cs))
        d, hes = jvp(f_diff, [clusters], [jnp.ones(shape=clusters.shape)])
        new_cluster = clusters - d / hes
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    return (jax.lax.while_loop(cond, body, (0, float("inf"), clusters)))[2]
