# Originally based on
# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/lstm_objective.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import tensorflow as tf


def lstm_model(weight, bias, hidden, cell, inp):
    gates = tf.concat((inp, hidden, inp, hidden), 0) * weight + bias
    forget, ingate, outgate, change = tf.split(gates, 4)

    cell = cell * tf.sigmoid(forget) + tf.sigmoid(ingate) * tf.tanh(change)
    hidden = tf.sigmoid(outgate) * tf.tanh(cell)

    return hidden, cell


def predict(w, w2, state, x):
    x = x * w2[0]

    new_state = []
    for i in range(0, state.shape[0], 2):
        hidden, cell = lstm_model(w[i], w[i + 1], state[i], state[i + 1], x)
        x = hidden
        new_state += [hidden, cell]

    return x * w2[1] + w2[2], tf.stack(new_state)


def lstm_objective(main_params, extra_params, state, sequence):
    """The average loss of the LSTM across a sequence of inputs."""

    def step(acc, elems):
        state, total = acc
        inp, ygold = elems
        ypred, state = predict(main_params, extra_params, state, inp)
        ynorm = ypred - tf.math.log(tf.reduce_sum(tf.exp(ypred)) + 2)
        return state, total + tf.reduce_sum(ygold * ynorm)

    _, total = tf.foldl(
        step,
        (sequence[:-1], sequence[1:]),
        initializer=(state, tf.constant(0.0, dtype=tf.float64)),
    )

    count = (sequence.shape[0] - 1) * sequence.shape[1]
    return -total / count
