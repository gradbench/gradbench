# Originally based on
# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowCommon/hand_objective.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import numpy as np
import tensorflow as tf


def pose_param_indices(n_theta):
    """Where each pose parameter comes from, as an index into `theta` extended
    with a zero (at index `n_theta`) and a one (at index `n_theta + 1`).

    The pose parameters are (one row each): global rotation, scale, global
    translation, two unused rows, and then four rows for each of five fingers,
    where only the first one or two components of the first three rows are
    nonzero.
    """
    zero, one = n_theta, n_theta + 1
    rows = [[0, 1, 2], [one] * 3, [3, 4, 5], [zero] * 3, [zero] * 3]
    i = 6
    for _ in range(5):
        rows += [[i, i + 1, zero], [i + 2, zero, zero], [i + 3, zero, zero]]
        rows += [[zero] * 3]
        i += 4
    return np.array(rows, dtype=np.int32)


def to_pose_params(theta):
    padded = tf.concat([theta, tf.constant([0.0, 1.0], dtype=theta.dtype)], 0)
    return tf.gather(padded, pose_param_indices(theta.shape[0]))


def matrices(*entries):
    """Stack a row-major list of 3x3 batched entries into a [b, 3, 3] tensor."""
    return tf.reshape(tf.stack(entries, -1), [-1, 3, 3])


def euler_angles_to_rotation_matrix(xzy):
    tx, tz, ty = tf.unstack(xzy, axis=-1)
    ctx, stx = tf.cos(tx), tf.sin(tx)
    cty, sty = tf.cos(ty), tf.sin(ty)
    ctz, stz = tf.cos(tz), tf.sin(tz)
    one, zero = tf.ones_like(tx), tf.zeros_like(tx)

    Rx = matrices(one, zero, zero, zero, ctx, -stx, zero, stx, ctx)
    Ry = matrices(cty, zero, -sty, zero, one, zero, sty, zero, cty)
    Rz = matrices(ctz, -stz, zero, stz, ctz, zero, zero, zero, one)
    return Rz @ Ry @ Rx


def get_posed_relatives(pose_params, base_relatives):
    R = euler_angles_to_rotation_matrix(pose_params[3:])
    # Embed every R in a 4x4 homogeneous transformation.
    tr = tf.pad(R, [[0, 0], [0, 1], [0, 1]]) + tf.constant(
        np.diag([0.0, 0.0, 0.0, 1.0]), dtype=R.dtype
    )
    return base_relatives @ tr


def relatives_to_absolutes(relatives, parents):
    absolutes = []
    for i, parent in enumerate(parents):
        if parent == -1:
            absolutes.append(relatives[i])
        else:
            absolutes.append(absolutes[parent] @ relatives[i])
    return tf.stack(absolutes)


def angle_axis_to_rotation_matrix(angle_axis):
    sqnorm = tf.reduce_sum(angle_axis**2)
    small = sqnorm < 1e-8  # Norm less than 1e-4.

    # Avoid division by zero (and NaN gradients) in the branch not taken.
    n = tf.sqrt(tf.where(small, tf.ones_like(sqnorm), sqnorm))
    x, y, z = tf.unstack(angle_axis / n)
    s, c = tf.sin(n), tf.cos(n)
    R = matrices(
        x * x + (1 - x * x) * c,
        x * y * (1 - c) - z * s,
        x * z * (1 - c) + y * s,
        x * y * (1 - c) + z * s,
        y * y + (1 - y * y) * c,
        y * z * (1 - c) - x * s,
        x * z * (1 - c) - y * s,
        z * y * (1 - c) + x * s,
        z * z + (1 - z * z) * c,
    )[0]
    return tf.where(small, tf.eye(3, dtype=R.dtype), R)


def apply_global_transform(pose_params, positions):
    R = angle_axis_to_rotation_matrix(pose_params[0]) * pose_params[1]
    return positions @ tf.transpose(R) + pose_params[2]


def get_skinned_vertex_positions(pose_params, model):
    relatives = get_posed_relatives(pose_params, model["base_relatives"])
    absolutes = relatives_to_absolutes(relatives, model["parents"])
    transforms = absolutes @ model["inverse_base_absolutes"]

    # Linear blend skinning: transform every vertex by every bone, then take
    # the weighted sum.
    positions = transforms @ tf.transpose(model["base_positions"])
    positions = tf.reduce_sum(positions * tf.transpose(model["weights"])[:, None, :], 0)
    positions = tf.transpose(positions[:3])

    if model["is_mirrored"]:
        positions *= tf.constant([-1.0, 1.0, 1.0], dtype=positions.dtype)

    return apply_global_transform(pose_params, positions)


def ht_objective(theta, model, points, correspondences):
    pose_params = to_pose_params(theta)
    vertex_positions = get_skinned_vertex_positions(pose_params, model)
    return points - tf.gather(vertex_positions, correspondences)


def ht_objective_complicated(theta, us, model, points, correspondences):
    pose_params = to_pose_params(theta)
    vertex_positions = get_skinned_vertex_positions(pose_params, model)

    # Every point corresponds to a point on a triangle of the hand mesh,
    # expressed in barycentric coordinates.
    triangles = tf.gather(model["triangles"], correspondences)
    corners = tf.gather(vertex_positions, triangles)  # [npts, 3, 3]
    us = tf.concat([us, 1.0 - tf.reduce_sum(us, 1, keepdims=True)], 1)
    hand_points = tf.reduce_sum(us[:, :, None] * corners, 1)

    return points - hand_points
