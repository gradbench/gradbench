from jax import grad

def square(x):
    return x * x


def double(x):
    gradient = grad(square)
    return gradient(x)
