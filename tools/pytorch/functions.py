def double(x):
    y = x * x
    y.backward()
    return x.grad
