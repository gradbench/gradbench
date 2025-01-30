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
def direction(input):
    k, clusters, features = input
    f_diff = grad(lambda cs: cost(features, cs))
    d, hes = jvp(f_diff, [clusters], [jnp.ones(shape=clusters.shape)])
    return d / hes
