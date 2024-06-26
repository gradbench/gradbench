def square(x):
    return x * x


def double(x):
    y = square(x)
    y.backward()
    return x.grad