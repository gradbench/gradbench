import tensorflow as tf
from gradbench import wrap


def to_tensor(x):
    return tf.Variable(x, dtype=tf.float64)


@wrap.function(pre=to_tensor, post=lambda x: x.numpy())
def square(x):
    return x * x


@wrap.function(pre=to_tensor, post=lambda x: x.numpy())
def double(x):
    with tf.GradientTape() as tape:
        y = square(x)

    return tape.gradient(y, x)
