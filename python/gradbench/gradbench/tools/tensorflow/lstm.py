# Originally based on
# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphLSTM.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import tensorflow as tf
from gradbench import wrap
from gradbench.tools.tensorflow.lstm_objective import lstm_objective


@tf.function(jit_compile=True)
def lstm_value(main_params, extra_params, state, sequence):
    return lstm_objective(main_params, extra_params, state, sequence)


@tf.function(jit_compile=True)
def lstm_gradient(main_params, extra_params, state, sequence):
    with tf.GradientTape() as tape:
        tape.watch((main_params, extra_params))
        obj = lstm_objective(main_params, extra_params, state, sequence)
    grads = tape.gradient(obj, (main_params, extra_params))
    return tf.concat([tf.reshape(g, [-1]) for g in grads], 0)


def prepare(input):
    return tuple(
        tf.constant(input[name], dtype=tf.float64)
        for name in ["main_params", "extra_params", "state", "sequence"]
    )


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    return lstm_value(*input)


@wrap.multiple_runs(pre=prepare, post=lambda g: g.numpy().tolist())
def jacobian(input):
    return lstm_gradient(*input)
