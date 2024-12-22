from functools import partial

import numpy as np
import torch
from torch.autograd.functional import vhp, vjp

from gradbench import wrap


def all_pairs_norm(a, b):
    a_sqr = (a**2).sum(1)[None, :]
    b_sqr = (b**2).sum(1)[:, None]
    diff = torch.matmul(b, a.T)
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def prepare_input(input):
    k = np.int64(input["k"])
    features = torch.tensor(np.array(input["points"], dtype=np.float32))
    return k, features


@wrap.multiple_runs(
    runs=lambda x: x["runs"], pre=prepare_input, post=lambda x: x.tolist()
)
def kmeans(input):
    k, features = input
    max_iter = 10
    tolerance = 1.0
    clusters = torch.flip(features[-int(k) :], (0,))
    t = 0
    converged = False
    while not converged and t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.0))
        _, hes = vhp(partial(cost, features), clusters, v=torch.ones_like(clusters))

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1
    return clusters
