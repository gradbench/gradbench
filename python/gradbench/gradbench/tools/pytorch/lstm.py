# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/PyTorch/PyTorchLSTM.py

"""
Changes Made:
- Added two functions to create a PyTorchHand object and call calculate_objective and calculate_jacobian
"""

import numpy as np
import torch

from gradbench import wrap
from gradbench.adbench.itest import ITest
from gradbench.adbench.lstm_data import LSTMInput, LSTMOutput
from gradbench.tools.pytorch.lstm_objective import lstm_objective
from gradbench.tools.pytorch.utils import to_torch_tensors, torch_jacobian


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

    def calculate_objective(self):
        self.objective = lstm_objective(*self.inputs, *self.params)

    def calculate_jacobian(self):
        self.objective, self.gradient = torch_jacobian(
            lstm_objective, self.inputs, self.params
        )


def prepare_input(input):
    py = PyTorchLSTM()
    py.prepare(LSTMInput.from_dict(input))
    return py


def objective_output(output):
    return float(output.detach().flatten().numpy()[0])


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
