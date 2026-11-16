# Originally based on
# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphBA.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import numpy as np
import tensorflow as tf
from gradbench import wrap
from gradbench.adbench.defs import BA_NCAMPARAMS
from gradbench.tools.tensorflow.ba_objective import compute_reproj_err, compute_w_err


@tf.function(jit_compile=True)
def ba_objective(cams, X, w, feats, obs):
    reproj_err = compute_reproj_err(
        tf.gather(cams, obs[:, 0]), tf.gather(X, obs[:, 1]), w, feats
    )
    return reproj_err, compute_w_err(w)


@tf.function(jit_compile=True)
def ba_jacobian(cams, X, w, feats, obs):
    """Compute the sparse Jacobian in compressed sparse row format."""
    n, m, p = cams.shape[0], X.shape[0], w.shape[0]
    cam_idx, pt_idx = obs[:, 0], obs[:, 1]

    # Each reprojection error depends on exactly one camera, one point, and
    # one weight, so its Jacobian is a dense 2x15 block per observation.
    params = tf.concat([tf.gather(cams, cam_idx), tf.gather(X, pt_idx), w[:, None]], 1)
    with tf.GradientTape() as tape:
        tape.watch(params)
        reproj_err = compute_reproj_err(
            params[:, :BA_NCAMPARAMS],
            params[:, BA_NCAMPARAMS:-1],
            params[:, -1],
            feats,
        )
    reproj_jac = tape.batch_jacobian(reproj_err, params)

    # Each weight error depends only on the corresponding weight.
    with tf.GradientTape() as tape:
        tape.watch(w)
        w_err = compute_w_err(w)
    w_jac = tape.gradient(w_err, w)

    block_size = BA_NCAMPARAMS + 3 + 1
    w_col = BA_NCAMPARAMS * n + 3 * m + tf.range(p)
    block_cols = tf.concat(
        [
            BA_NCAMPARAMS * cam_idx[:, None] + tf.range(BA_NCAMPARAMS),
            BA_NCAMPARAMS * n + 3 * pt_idx[:, None] + tf.range(3),
            w_col[:, None],
        ],
        1,
    )
    # Both rows of a block have the same columns.
    block_cols = tf.repeat(block_cols, 2, axis=0)

    rows = tf.concat(
        [
            block_size * tf.range(2 * p + 1),
            2 * block_size * p + tf.range(1, p + 1),
        ],
        0,
    )
    cols = tf.concat([tf.reshape(block_cols, [-1]), w_col], 0)
    vals = tf.concat([tf.reshape(reproj_jac, [-1]), w_jac], 0)
    return rows, cols, vals


def prepare(input):
    n, m, p = input["n"], input["m"], input["p"]
    cams = np.tile(input["cam"], (n, 1))
    X = np.tile(input["x"], (m, 1))
    w = np.full(p, input["w"])
    feats = np.tile(input["feat"], (p, 1))
    obs = np.stack([np.arange(p) % n, np.arange(p) % m], 1)

    args = tuple(tf.constant(a, dtype=tf.float64) for a in (cams, X, w, feats))
    args += (tf.constant(obs, dtype=tf.int32),)
    return args


def objective_output(errors):
    reproj_err, w_err = errors
    return {
        "reproj_error": {
            "elements": reproj_err[0].numpy().tolist(),
            "repeated": reproj_err.shape[0],
        },
        "w_err": {"element": float(w_err[0]), "repeated": w_err.shape[0]},
    }


def jacobian_output(jacobian):
    def dedup(A):
        return tf.concat([A[:30], A[-1:]], 0).numpy().tolist()

    rows, cols, vals = jacobian
    return {"rows": dedup(rows), "cols": dedup(cols), "vals": dedup(vals)}


@wrap.multiple_runs(pre=prepare, post=objective_output)
def objective(input):
    return ba_objective(*input)


@wrap.multiple_runs(pre=prepare, post=jacobian_output)
def jacobian(input):
    return ba_jacobian(*input)
