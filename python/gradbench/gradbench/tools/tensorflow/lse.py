import numpy as np
import tensorflow as tf
from gradbench import wrap


def logsumexp(x):
    max_val = tf.reduce_max(x)
    return max_val + tf.math.log(tf.reduce_sum(tf.exp(x - max_val)))


def prepare_input(input):
    return tf.Variable(np.array(input["x"], dtype=np.float64), dtype=tf.float64)


@wrap.function(pre=prepare_input, post=lambda x: float(x))
def primal(x):
    return logsumexp(x)


@wrap.function(pre=prepare_input, post=lambda x: x.numpy().tolist())
def gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = logsumexp(x)

    grad = tape.gradient(y, x)
    return grad
