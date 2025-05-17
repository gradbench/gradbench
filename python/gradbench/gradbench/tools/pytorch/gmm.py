from dataclasses import dataclass

import torch
from gradbench import wrap
from gradbench.tools.pytorch.gmm_objective import gmm_objective


@dataclass
class Independent:
    alpha: torch.Tensor
    mu: torch.Tensor
    q: torch.Tensor
    l: torch.Tensor


@dataclass
class Input(Independent):
    d: int
    k: int
    n: int
    x: torch.Tensor
    m: torch.Tensor
    gamma: float


def prepare(input):
    def tensor(data, requires_grad=False):
        return torch.tensor(data, dtype=torch.float64, requires_grad=requires_grad)

    def active_tensor(data):
        return tensor(data, requires_grad=True)

    def torchify(*, d, k, n, x, m, gamma, alpha, mu, q, l, **_):
        return Input(
            d=d,
            k=k,
            n=n,
            x=tensor(x),
            m=tensor(m),
            gamma=gamma,
            alpha=active_tensor(alpha),
            mu=active_tensor(mu),
            q=active_tensor(q),
            l=active_tensor(l),
        )

    return torchify(**input)


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    return gmm_objective(**vars(input))


def postprocess(input):
    def listify(*, alpha, mu, q, l, **_):
        return {
            "alpha": alpha.grad.tolist(),
            "mu": mu.grad.tolist(),
            "q": q.grad.tolist(),
            "l": l.grad.tolist(),
        }

    return listify(**vars(input))


@wrap.multiple_runs(pre=prepare, post=postprocess)
def jacobian(input):
    gmm_objective(**vars(input)).backward()
    return input
