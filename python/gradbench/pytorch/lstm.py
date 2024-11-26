# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/PyTorch/PyTorchLSTM.py

"""
Changes Made:
- Added two functions to create a PyTorchHand object and call calculate_objective and calculate_jacobian
"""

import numpy as np
import torch

from gradbench.adbench.itest import ITest
from gradbench.adbench.lstm_data import LSTMInput, LSTMOutput
from gradbench.pytorch.lstm_objective import lstm_objective
from gradbench.pytorch.utils import to_torch_tensors, torch_jacobian
from gradbench.wrap_module import wrap


class PyTorchLSTM(ITest):
    """Test class for LSTM diferentiation by PyTorch."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.inputs = to_torch_tensors(
            (input.main_params, input.extra_params), grad_req=True
        )

        self.params = to_torch_tensors((input.state, input.sequence))
        self.gradient = torch.empty(0)
        self.objective = torch.zeros(1)

    def output(self):
        """Returns calculation result."""

        return LSTMOutput(self.objective.item(), self.gradient.numpy())

    def calculate_objective(self, times):
        """Calculates objective function many times."""

        for i in range(times):
            self.objective = lstm_objective(*self.inputs, *self.params)

    def calculate_jacobian(self, times):
        """Calculates objective function jacobian many times."""

        for i in range(times):
            self.objective, self.gradient = torch_jacobian(
                lstm_objective, self.inputs, self.params
            )


def prepare_input(input):
    return LSTMInput.from_dict(input)


def objective_output(output):
    return output.detach().flatten().numpy().tolist()


def jacobian_output(output):
    return output.tolist()


@wrap(prepare_input, objective_output)
def calculate_objectiveLSTM(input):
    py = PyTorchLSTM()
    py.prepare(input)
    py.calculate_objective(1)
    return py.objective


@wrap(prepare_input, jacobian_output)
def calculate_jacobianLSTM(input):
    py = PyTorchLSTM()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.gradient
