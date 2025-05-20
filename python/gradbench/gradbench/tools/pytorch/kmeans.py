from functools import partial

import numpy as np
import torch
from gradbench import wrap
from torch.autograd.functional import vhp, vjp


def all_pairs_norm(a, b):
    a_sqr = (a**2).sum(1)[None, :]
    b_sqr = (b**2).sum(1)[:, None]
    diff = torch.matmul(b, a.T)
    return a_sqr + b_sqr - 2 * diff


def costfun(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def prepare_input(input):
    points = torch.tensor(np.array(input["points"], dtype=np.float64))
    centroids = torch.tensor(np.array(input["centroids"], dtype=np.float64))
    return points, centroids


@wrap.multiple_runs(pre=prepare_input, post=float)
def cost(input):
    points, centroids = input
    return costfun(points, centroids)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def dir(input):
    points, centroids = input
    _, jac = vjp(partial(costfun, points), centroids, v=torch.tensor(1.0))
    _, hes = vhp(partial(costfun, points), centroids, v=torch.ones_like(centroids))

    return jac / hes
