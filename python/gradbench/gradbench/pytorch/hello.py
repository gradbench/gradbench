import torch

from gradbench import wrap


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def square(x):
    return x * x


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def double(x):
    y = square(x)
    y.backward()
    return x.grad
