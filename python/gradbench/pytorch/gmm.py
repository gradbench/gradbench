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

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/PyTorch/PyTorchGMM.py

"""
Changes Made:
- Added two functions to create a PyTorchGMM object and call calculate_objective and calculate_jacobian
- Added function to create GMMInput object
- Import a decorator to convert input and outputs to necessary types
"""

import time

import numpy as np
import torch

from gradbench.adbench.defs import Wishart
from gradbench.adbench.gmm_data import GMMInput
from gradbench.adbench.itest import ITest
from gradbench.pytorch.gmm_objective import gmm_objective
from gradbench.pytorch.utils import to_torch_tensor, to_torch_tensors, torch_jacobian
from gradbench.wrap_module import wrap


class PyTorchGMM(ITest):
    """Test class for GMM differentiation by PyTorch."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.inputs = to_torch_tensors(
            (input.alphas, input.means, input.icf), grad_req=True
        )

        self.params = to_torch_tensors((input.x, input.wishart.gamma, input.wishart.m))

        self.objective = torch.zeros(1)
        self.gradient = torch.empty(0)

    def output(self):
        """Returns calculation result."""

        return GMMOutput(self.objective.item(), self.gradient.numpy())

    def calculate_objective(self, times):
        """Calculates objective function many times."""

        for i in range(times):
            self.objective = gmm_objective(*self.inputs, *self.params)

    def calculate_jacobian(self, times):
        """Calculates objective function jacobian many times."""

        for i in range(times):
            self.objective, self.gradient = torch_jacobian(
                gmm_objective, self.inputs, self.params
            )


def prepare_input(input):
    py = PyTorchGMM()
    py.prepare(
        GMMInput(
            input["alpha"],
            input["means"],
            input["icf"],
            input["x"],
            Wishart(input["gamma"], input["m"]),
        )
    )
    return py, input["runs"]


def evaluate_times(times):
    return [{"name": "evaluate", "nanoseconds": time} for time in times]


def unwrap_objective(output):
    py, times = output
    return py.objective.tolist(), evaluate_times(times)


def unwrap_jacobian(output):
    py, times = output
    return py.gradient.tolist(), evaluate_times(times)


@wrap(prepare_input, unwrap_objective)
def objective(input):
    py, runs = input
    times = runs * [None]
    for i in range(runs):
        start = time.perf_counter_ns()
        py.calculate_jacobian(runs)
        end = time.perf_counter_ns()
        times[i] = end - start
    return py, times


@wrap(prepare_input, unwrap_jacobian)
def jacobian(input):
    py, runs = input
    times = runs * [None]
    for i in range(runs):
        start = time.perf_counter_ns()
        py.calculate_jacobian(runs)
        end = time.perf_counter_ns()
        times[i] = end - start
    return py, times
