import torch
from wrap_module import wrap


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


@wrap(to_tensor, lambda x: x.item())
def square(x):
    return x * x


@wrap(to_tensor, lambda x: x.item())
def double(x):
    y = square(x)
    y.backward()
    return x.grad
