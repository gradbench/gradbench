from autograd import grad

from gradbench import wrap


@wrap.function(pre=lambda x: x * 1.0, post=id)
def square(x):
    return x * x


@wrap.function(pre=lambda x: x * 1.0, post=id)
def double(x):
    gradient = grad(square)
    return gradient(x)
