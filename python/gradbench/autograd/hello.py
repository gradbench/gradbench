from autograd import grad

from gradbench.wrap_module import wrap


@wrap(lambda x: x * 1.0, lambda x: x)
def square(x):
    return x * x


@wrap(lambda x: x * 1.0, lambda x: x)
def double(x):
    y = grad(square)
    return y(x)
