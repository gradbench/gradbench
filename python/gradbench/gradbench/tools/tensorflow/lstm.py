# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphLSTM.py

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()  # turn eager execution off

from gradbench import wrap
from gradbench.adbench.itest import ITest
from gradbench.adbench.lstm_data import LSTMInput
from gradbench.tools.tensorflow.lstm_objective import lstm_objective
from gradbench.tools.tensorflow.utils import flatten, to_tf_tensor


class TensorflowLSTM(ITest):
    """Test class for LSTM diferentiation by Tensorflow using computational
    graphs."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before any others."""

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            self.main_params = input.main_params
            self.extra_params = input.extra_params
            self.state = to_tf_tensor(input.state)
            self.sequence = to_tf_tensor(input.sequence)

            self.prepare_operations()

        self.session = tf.compat.v1.Session(graph=graph)
        self.first_running()

        self.gradient = np.zeros(0)
        self.objective = np.zeros(1)

    def prepare_operations(self):
        """Prepares computational graph for needed operations."""

        self.main_params_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.main_params.shape
        )

        self.extra_params_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.extra_params.shape
        )

        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(self.main_params_placeholder)
            grad_tape.watch(self.extra_params_placeholder)

            self.objective_operation = lstm_objective(
                self.main_params_placeholder,
                self.extra_params_placeholder,
                self.state,
                self.sequence,
            )

        J = grad_tape.gradient(
            self.objective_operation,
            (self.main_params_placeholder, self.extra_params_placeholder),
        )

        self.gradient_operation = tf.concat([flatten(d) for d in J], 0)

        self.feed_dict = {
            self.main_params_placeholder: self.main_params,
            self.extra_params_placeholder: self.extra_params,
        }

    def first_running(self):
        """Performs the first session running."""

        self.session.run(self.objective_operation, feed_dict=self.feed_dict)

        self.session.run(self.gradient_operation, feed_dict=self.feed_dict)

    def calculate_objective(self):
        self.objective = self.session.run(
            self.objective_operation, feed_dict=self.feed_dict
        )

    def calculate_jacobian(self):
        self.gradient = self.session.run(
            self.gradient_operation, feed_dict=self.feed_dict
        )


def prepare_input(input):
    py = TensorflowLSTM()
    py.prepare(LSTMInput.from_dict(input))
    return py


def objective_output(output):
    return float(output.flatten()[0])


def jacobian_output(output):
    return output.tolist()


@wrap.multiple_runs(pre=prepare_input, post=objective_output)
def objective(py):
    py.calculate_objective()
    return py.objective


@wrap.multiple_runs(pre=prepare_input, post=jacobian_output)
def jacobian(py):
    py.calculate_jacobian()
    return py.gradient
