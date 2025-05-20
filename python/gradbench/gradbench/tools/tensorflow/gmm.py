# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowGraph/TensorflowGraphGMM.py

"""
Changes made:
  * Use simplified ITest interface.
  * Add jacobian/objective top level functions.
"""

import numpy as np
import tensorflow as tf
from gradbench import wrap
from gradbench.adbench.defs import Wishart
from gradbench.adbench.gmm_data import GMMInput
from gradbench.adbench.itest import ITest
from gradbench.tools.tensorflow.gmm_objective import gmm_objective
from gradbench.tools.tensorflow.utils import flatten, to_tf_tensor
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()  # turn eager execution off


class TensorflowGMM(ITest):
    """Test class for GMM diferentiation by Tensorflow using computational
    graphs."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.alphas = input.alphas
        self.means = input.means
        self.icf = input.icf

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            self.x = to_tf_tensor(input.x)
            self.wishart_gamma = input.wishart.gamma
            self.wishart_m = input.wishart.m

            self.prepare_operations()

        self.session = tf.compat.v1.Session(graph=graph)
        self.first_running()

        self.objective = np.zeros(1)
        self.gradient = np.zeros(0)

    def prepare_operations(self):
        """Prepares computational graph for needed operations."""

        self.alphas_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.alphas.shape
        )

        self.means_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.means.shape
        )

        self.icf_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=self.icf.shape
        )

        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(self.alphas_placeholder)
            grad_tape.watch(self.means_placeholder)
            grad_tape.watch(self.icf_placeholder)

            self.objective_operation = gmm_objective(
                self.alphas_placeholder,
                self.means_placeholder,
                self.icf_placeholder,
                self.x,
                self.wishart_gamma,
                self.wishart_m,
            )

        J = grad_tape.gradient(
            self.objective_operation,
            (self.alphas_placeholder, self.means_placeholder, self.icf_placeholder),
        )

        self.gradient_operation = tf.concat([flatten(d) for d in J], 0)

        self.feed_dict = {
            self.alphas_placeholder: self.alphas,
            self.means_placeholder: self.means,
            self.icf_placeholder: self.icf,
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
    py = TensorflowGMM()
    py.prepare(
        GMMInput(
            np.array(input["alpha"]),
            np.array(input["means"]),
            np.array(input["icf"]),
            np.array(input["x"]),
            Wishart(input["gamma"], input["m"]),
        )
    )
    return py


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def jacobian(py):
    py.calculate_jacobian()
    return py.gradient


@wrap.multiple_runs(pre=prepare_input, post=float)
def objective(py):
    py.calculate_objective()
    return py.objective
