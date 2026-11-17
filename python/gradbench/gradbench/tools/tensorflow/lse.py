import tensorflow as tf
from gradbench import wrap


@tf.function
def logsumexp(x):
    return tf.reduce_logsumexp(x)


@tf.function
def logsumexp_gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.reduce_logsumexp(x)
    return tape.gradient(y, x)


def prepare(input):
    return tf.constant(input["x"], dtype=tf.float64)


@wrap.multiple_runs(pre=prepare, post=float)
def primal(x):
    return logsumexp(x)


@wrap.multiple_runs(pre=prepare, post=lambda g: g.numpy().tolist())
def gradient(x):
    return logsumexp_gradient(x)
