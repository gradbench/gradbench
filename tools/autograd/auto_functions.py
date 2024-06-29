from autograd import grad


def square(x):
    return x * x


def double(x):
    y = grad(square)
    return y(x)
