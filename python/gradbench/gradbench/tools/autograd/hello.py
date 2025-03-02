from autograd import grad
from gradbench import wrap


@wrap.function(pre=lambda x: x * 1.0, post=lambda x: x)
def square(x):
    return x * x


@wrap.function(pre=lambda x: x * 1.0, post=lambda x: x)
def double(x):
    gradient = grad(square)
    return gradient(x)
