import tensorflow as tf

from gradbench.wrap_module import wrap


def to_tensor(x):
    return tf.Variable(x, dtype=tf.float64)


@wrap(to_tensor, lambda x: x.numpy())
def square(x):
    return x * x


@wrap(to_tensor, lambda x: x.numpy())
def double(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = square(x)

    grad = tape.gradient(y, x)
    return grad
