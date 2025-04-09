import numpy as np
import torch
from gradbench import wrap
from torch.autograd.functional import vjp


def logsumexp(x):
    max_val = torch.max(x)
    return max_val + torch.log(torch.sum(torch.exp(x - max_val)))


def prepare_input(input):
    return torch.tensor(np.array(input["x"], dtype=np.float64))


@wrap.multiple_runs(pre=prepare_input, post=lambda x: float(x))
def primal(input):
    return logsumexp(input)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def gradient(input):
    return vjp(logsumexp, input, v=torch.tensor(1.0))[1]


# Force graph compilation.
vjp(logsumexp, torch.tensor([1.0, 2.0, 3.0]), v=torch.tensor(1.0))[1]
