# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/PyTorch/PyTorchGMM.py
#
# Changes: adapted for Autograd.

import autograd
import numpy as np

from gradbench import wrap
from gradbench.adbench.defs import Wishart
from gradbench.adbench.gmm_data import GMMInput
from gradbench.adbench.itest import ITest
from gradbench.autograd.gmm_objective import gmm_objective


def gmm_objective_wrapper(params, x, wishart_gamma, wishart_m):
    return gmm_objective(params[0], params[1], params[2], x, wishart_gamma, wishart_m)


class AutogradGMM(ITest):
    """Test class for GMM differentiation by Autograd."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.inputs = (input.alphas, input.means, input.icf)

        self.params = (input.x, input.wishart.gamma, input.wishart.m)

        self.objective = np.zeros(1)
        self.gradient = np.empty(0)

    def output(self):
        """Returns calculation result."""

        return GMMOutput(self.objective.item(), self.gradient.numpy())

    def calculate_objective(self, times):
        """Calculates objective function many times."""

        for i in range(times):
            self.objective = gmm_objective_wrapper(self.inputs, *self.params)

    def calculate_jacobian(self, times):
        """Calculates objective function jacobian many times."""

        for i in range(times):
            self.gradient = autograd.grad(gmm_objective_wrapper)(
                self.inputs, *self.params
            )


def prepare_input(input):
    py = AutogradGMM()
    py.prepare(
        GMMInput(
            np.array(input["alpha"]),
            np.array(input["means"]),
            np.array(input["icf"]),
            np.array(input["x"]),
            Wishart(np.array(input["gamma"]), np.array(input["m"])),
        )
    )
    return py


@wrap.multiple_runs(
    runs=lambda x: x["runs"], pre=prepare_input, post=lambda x: x.tolist()
)
def objective(py):
    py.calculate_objective(1)
    return py.objective


def unpack_jacobian(gradient):
    return np.concatenate(
        (gradient[0], gradient[1].flatten(), gradient[2].flatten())
    ).tolist()


@wrap.multiple_runs(runs=lambda x: x["runs"], pre=prepare_input, post=unpack_jacobian)
def jacobian(py):
    py.calculate_jacobian(1)
    return py.gradient
