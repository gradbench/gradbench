# Originally based on
# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphHand.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import tensorflow as tf
from gradbench import wrap
from gradbench.adbench.ht_data import HandInput
from gradbench.tools.tensorflow.ht_objective import (
    ht_objective,
    ht_objective_complicated,
)

# We use XLA (jit_compile) to fuse the many small operations in these
# functions, which makes them significantly faster.


@tf.function(jit_compile=True)
def ht_value(theta, us, **params):
    if us is None:
        return ht_objective(theta, **params)
    else:
        return ht_objective_complicated(theta, us, **params)


def forward_jacobian(f, x, tangents):
    """The Jacobian-vector products of `f` at `x` for every tangent in the
    batch `tangents`, computed with forward mode and vectorised over the batch.
    The result has the tangents as columns."""

    def jvp(tangent):
        with tf.autodiff.ForwardAccumulator(x, tangent) as acc:
            out = f(x)
        return tf.reshape(acc.jvp(out), [-1])

    return tf.transpose(tf.vectorized_map(jvp, tangents))


@tf.function(jit_compile=True)
def ht_jacobian(theta, us, **params):
    theta_tangents = tf.eye(theta.shape[0], dtype=theta.dtype)

    if us is None:
        return forward_jacobian(
            lambda theta: ht_objective(theta, **params), theta, theta_tangents
        )

    J_theta = forward_jacobian(
        lambda theta: ht_objective_complicated(theta, us, **params),
        theta,
        theta_tangents,
    )
    # The Jacobian with respect to `us` is block diagonal, as each point
    # depends only on its own two `us`. This means we can compute the nonzero
    # entries for all points simultaneously with just two Jacobian-vector
    # products.
    us_tangents = tf.eye(2, dtype=us.dtype)[:, None, :] * tf.ones_like(us)
    J_us = forward_jacobian(
        lambda us: ht_objective_complicated(theta, us, **params), us, us_tangents
    )
    return tf.concat([J_us, J_theta], 1)


def prepare(input):
    input = HandInput.from_dict(input)
    model = input.data.model

    def tensor(x):
        return tf.constant(x, dtype=tf.float64)

    return {
        "theta": tensor(input.theta),
        "us": tensor(input.us) if len(input.us) > 0 else None,
        "model": {
            "parents": tuple(model.parents.tolist()),
            "base_relatives": tensor(model.base_relatives),
            "inverse_base_absolutes": tensor(model.inverse_base_absolutes),
            "base_positions": tensor(model.base_positions),
            "weights": tensor(model.weights),
            "triangles": tf.constant(model.triangles, dtype=tf.int32),
            "is_mirrored": model.is_mirrored,
        },
        "points": tensor(input.data.points),
        "correspondences": tf.constant(input.data.correspondences, dtype=tf.int32),
    }


@wrap.multiple_runs(pre=prepare, post=lambda x: x.numpy().flatten().tolist())
def objective(input):
    return ht_value(**input)


@wrap.multiple_runs(pre=prepare, post=lambda x: x.numpy().tolist())
def jacobian(input):
    return ht_jacobian(**input)
