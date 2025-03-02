import numpy as np
from gradbench import wrap
from mygrad import tensor as mg_tensor


def to_tensor(x):
    return mg_tensor(x, dtype=np.float64)


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def square(x):
    return x * x


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def double(x):
    y = square(x)
    y.backward()
    return x.grad
