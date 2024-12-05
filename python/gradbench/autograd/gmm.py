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

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Autograd/Autograd_gmm_full.py

"""
Changes Made:
- Used pieces of original but adjusted the format to follow a class base structure similar to pytorch/gmm.py
- Adjusted jacobian calculation to align with how it is computed in pytorch/gmm.py
"""

import autograd.numpy as np
from autograd import value_and_grad

# from autograd import value_and_grad
from gradbench.adbench.itest import ITest
from gradbench.autograd.gmm_objective import gmm_objective
from gradbench.wrap_module import wrap


class AutogradGMM(ITest):
    def prepare(self, inputs):
        self.inputs = inputs
        self.objective = None
        self.gradient = None

    def output(self):
        return self.objective, self.gradient
        # return

    def calculate_objective(self, times):
        for i in range(times):
            self.objective = gmm_objective(
                self.inputs[0],
                self.inputs[1],
                self.inputs[2],
                self.inputs[3],
                self.inputs[4],
                self.inputs[5],
            )

    def calculate_jacobian(self, times):
        def fixed_objective(alphas, means, icf):
            return gmm_objective(
                alphas, means, icf, self.inputs[3], self.inputs[4], self.inputs[5]
            )

        objective_and_grad = value_and_grad(fixed_objective, argnum=(0, 1, 2))

        for _ in range(times):
            _, gradients = objective_and_grad(
                self.inputs[0],  # alphas
                self.inputs[1],  # means
                self.inputs[2],  # icf
            )

            g_alphas, g_means, g_icf = gradients
            # Flatten and concatenate gradients to match the expected shape of G

            self.gradient = np.concatenate(
                [
                    g_alphas.flatten(),  # Shape (K,)
                    g_means.flatten(),  # Shape (K×D,)
                    g_icf.flatten(),  # Shape (K×(D + D(D-1)/2))
                ]
            )


def gmm_objective_wrapper(alphas, means, icf, x, wishart_gamma, wishart_m):
    return gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m)


def prepare_input(input):
    return (
        np.array(input["alpha"]),
        np.array(input["means"]),
        np.array(input["icf"]),
        np.array(input["x"]),
        np.array(input["gamma"]),
        np.array(input["m"]),
    )


@wrap(prepare_input, lambda x: x.tolist())
def calculate_jacobianGMM(input):
    py = AutogradGMM()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.gradient


@wrap(prepare_input, lambda x: x.tolist())
def calculate_objectiveGMM(input):
    py = AutogradGMM()
    py.prepare(input)
    py.calculate_objective(1)
    return py.objective
