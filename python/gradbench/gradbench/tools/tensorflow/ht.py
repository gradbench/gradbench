# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphHand.py

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()  # turn eager execution off

from gradbench import wrap
from gradbench.adbench.ht_data import HandInput
from gradbench.adbench.itest import ITest
from gradbench.tools.tensorflow.ht_objective import (
    ht_objective,
    ht_objective_complicated,
)
from gradbench.tools.tensorflow.utils import flatten, to_tf_tensor


class TensorflowHT(ITest):
    """Test class for HT diferentiation by Tensorflow using computational
    graphs."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before any others."""

        self.complicated = len(input.us) > 0
        self.nrows = 3 * len(input.data.correspondences)

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            if self.complicated:
                self.variables = np.append(input.us.flatten(), input.theta)
                self.params = (
                    input.data.model.bone_count,
                    input.data.model.parents,
                    to_tf_tensor(input.data.model.base_relatives),
                    to_tf_tensor(input.data.model.inverse_base_absolutes),
                    to_tf_tensor(input.data.model.base_positions),
                    tf.transpose(to_tf_tensor(input.data.model.weights)),
                    input.data.model.is_mirrored,
                    to_tf_tensor(input.data.points),
                    input.data.correspondences,
                    input.data.model.triangles,
                )

                self.objective_function = ht_objective_complicated
                self.ncols = len(input.theta) + 2
            else:
                self.variables = input.theta
                self.params = (
                    input.data.model.bone_count,
                    input.data.model.parents,
                    to_tf_tensor(input.data.model.base_relatives),
                    to_tf_tensor(input.data.model.inverse_base_absolutes),
                    to_tf_tensor(input.data.model.base_positions),
                    tf.transpose(to_tf_tensor(input.data.model.weights)),
                    input.data.model.is_mirrored,
                    to_tf_tensor(input.data.points),
                    input.data.correspondences,
                )

                self.objective_function = ht_objective
                self.ncols = len(input.theta)

            self.prepare_operations()

        self.session = tf.compat.v1.Session(graph=graph)
        self.first_running()

        self.objective = None
        self.jacobian = None

    def prepare_operations(self):
        """Prepares computational graph for needed operations."""

        self.variables_holder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.variables.shape
        )

        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(self.variables_holder)
            self.objective_operation = self.objective_function(
                self.variables_holder, *self.params
            )

            self.objective_operation = flatten(self.objective_operation)

        self.jacobian_operation = grad_tape.jacobian(
            self.objective_operation, self.variables_holder, experimental_use_pfor=False
        )

        self.feed_dict = {self.variables_holder: self.variables}

    def first_running(self):
        """Performs the first session running."""

        self.session.run(self.objective_operation, feed_dict=self.feed_dict)

        self.session.run(self.jacobian_operation, feed_dict=self.feed_dict)

    def calculate_objective(self):
        self.objective = self.session.run(
            self.objective_operation, feed_dict=self.feed_dict
        )

    def calculate_jacobian(self):
        self.jacobian = self.session.run(
            self.jacobian_operation, feed_dict=self.feed_dict
        )

        if self.complicated:
            # Merging us part of jacobian to two columns.
            # Note: jacobian has the following structure:
            #
            #   [us_part theta_part]
            #
            # where us part is a block diagonal matrix with blocks of
            # size [3, 2]

            us_J = np.concatenate(
                [
                    self.jacobian[3 * i : 3 * i + 3, 2 * i : 2 * i + 2]
                    for i in range(self.nrows // 3)
                ],
                0,
            )

            us_count = 2 * self.nrows // 3
            theta_J = self.jacobian[:, us_count:]

            self.jacobian = np.concatenate((us_J, theta_J), 1)


def prepare_input(input):
    py = TensorflowHT()
    py.prepare(HandInput.from_dict(input))
    return py


def objective_output(output):
    return output.flatten().tolist()


def jacobian_output(output):
    return output.tolist()


@wrap.multiple_runs(pre=prepare_input, post=objective_output)
def objective(py):
    py.calculate_objective()
    return py.objective


@wrap.multiple_runs(pre=prepare_input, post=jacobian_output)
def jacobian(py):
    py.calculate_jacobian()
    return py.jacobian
