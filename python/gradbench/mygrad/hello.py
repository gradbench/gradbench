import numpy as np
from mygrad import tensor as mg_tensor

from gradbench.wrap_module import wrap


def to_tensor(x):
    return mg_tensor(x, dtype=np.float64)


@wrap(to_tensor, lambda x: x.item())
def square(x):
    return x * x


@wrap(to_tensor, lambda x: x.item())
def double(x):
    y = square(x)
    y.backward()
    return x.grad
