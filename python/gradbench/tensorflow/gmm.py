# Copyright (c) Microsoft Corporation.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/Tensorflow/TensorflowGMM.py

"""
Changes Made:
- Added two functions to create a TensorflowGMM object and call calculate_objective and calculate_jacobian
- Added function to create GMMInput object
- Import a decorator to convert input and outputs to necessary types
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from gradbench.adbench.defs import Wishart
from gradbench.adbench.gmm_data import GMMInput, GMMOutput
from gradbench.adbench.itest import ITest
from gradbench.tensorflow.gmm_objective import gmm_objective
from gradbench.tensorflow.utils import flatten, to_tf_tensor
from gradbench.wrap_module import wrap


class TensorflowGMM(ITest):
    """Test class for GMM differentiation by Tensorflow using eager
    execution."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.alphas = to_tf_tensor(input.alphas)
        self.means = to_tf_tensor(input.means)
        self.icf = to_tf_tensor(input.icf)

        self.x = to_tf_tensor(input.x)
        self.wishart_gamma = to_tf_tensor(input.wishart.gamma)
        self.wishart_m = to_tf_tensor(input.wishart.m)

        self.objective = tf.zeros(1)
        self.gradient = tf.zeros(0)

    def output(self):
        """Returns calculation result."""

        return GMMOutput(self.objective.numpy(), self.gradient.numpy())

    def calculate_objective(self, times):
        """Calculates objective function many times."""

        for _ in range(times):
            self.objective = gmm_objective(
                self.alphas,
                self.means,
                self.icf,
                self.x,
                self.wishart_gamma,
                self.wishart_m,
            )

    def calculate_jacobian(self, times):
        """Calculates objective function jacobian many times."""

        for _ in range(times):
            with tf.GradientTape(persistent=True) as t:
                t.watch(self.alphas)
                t.watch(self.means)
                t.watch(self.icf)

                self.objective = gmm_objective(
                    self.alphas,
                    self.means,
                    self.icf,
                    self.x,
                    self.wishart_gamma,
                    self.wishart_m,
                )

            J = t.gradient(self.objective, (self.alphas, self.means, self.icf))
            self.gradient = tf.concat([flatten(d) for d in J], 0)


def prepare_input(input):
    return GMMInput(
        input["alpha"],
        input["means"],
        input["icf"],
        input["x"],
        Wishart(input["gamma"], input["m"]),
    )


@wrap(prepare_input, lambda x: x.numpy().tolist())
def calculate_jacobianGMM(input):
    py = TensorflowGMM()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.gradient


@wrap(prepare_input, lambda x: x.numpy().tolist())
def calculate_objectiveGMM(input):
    py = TensorflowGMM()
    py.prepare(input)
    py.calculate_objective(1)
    return py.objective
