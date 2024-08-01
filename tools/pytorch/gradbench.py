import torch
from wrap_module import wrap


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


@wrap(to_tensor, lambda x: x.item())
def square(x):
    return x * x


@wrap(to_tensor, lambda x: x.item())
def double(x):
    # call the function without the decorator, could also write second function or call x*x here
    y = square.__wrapped__(x)
    y.backward()
    return x.grad
