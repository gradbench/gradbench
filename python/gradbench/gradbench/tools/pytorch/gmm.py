import torch
from gradbench import wrap
from gradbench.tools.pytorch.gmm_objective import gmm_objective


def prepare(input):
    def tensor(data, requires_grad=False):
        return torch.tensor(data, dtype=torch.float64, requires_grad=requires_grad)

    def active_tensor(data):
        return tensor(data, requires_grad=True)

    def torchify(*, d, k, n, x, m, gamma, alpha, mu, q, l, **_):
        return {
            "d": d,
            "k": k,
            "n": n,
            "x": tensor(x),
            "m": tensor(m),
            "gamma": gamma,
            "alpha": active_tensor(alpha),
            "mu": active_tensor(mu),
            "q": active_tensor(q),
            "l": active_tensor(l),
        }

    return torchify(**input)


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    return gmm_objective(**input)


def postprocess(input):
    def listify(*, alpha, mu, q, l, **_):
        return {
            "alpha": alpha.grad.tolist(),
            "mu": mu.grad.tolist(),
            "q": q.grad.tolist(),
            "l": l.grad.tolist(),
        }

    return listify(**input)


@wrap.multiple_runs(pre=prepare, post=postprocess)
def jacobian(input):
    def zero(grad):
        if grad is not None:
            grad.zero_()

    zero(input["alpha"].grad)
    zero(input["mu"].grad)
    zero(input["q"].grad)
    zero(input["l"].grad)
    gmm_objective(**input).backward()
    return input
