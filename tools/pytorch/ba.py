# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Changes Made:
- Added two functions to create a PyTorchBA object and call calculate_objective and calculate_jacobian
"""

import numpy as np
import torch
from ba_data import BAInput, BAOutput
from ba_objective import compute_reproj_err, compute_w_err
from ba_sparse_mat import BASparseMat
from itest import ITest
from utils import to_torch_tensor, to_torch_tensors, torch_jacobian


class PyTorchBA(ITest):
    """Test class for BA diferentiation by PyTorch."""

    def prepare(self, input):
        """Prepares calculating. This function must be run before
        any others."""

        self.p = len(input.obs)

        # we use tuple of tensors instead of multidimensional tensor
        # because torch doesn't differentiate by non-leaf tensors
        self.cams = tuple(to_torch_tensor(cam, grad_req=True) for cam in input.cams)

        self.x = tuple(to_torch_tensor(x, grad_req=True) for x in input.x)

        self.w = tuple(to_torch_tensor(w, grad_req=True) for w in input.w)

        self.obs = to_torch_tensor(input.obs, dtype=torch.int64)
        self.feats = to_torch_tensor(input.feats)

        self.reproj_error = torch.zeros(2 * self.p, dtype=torch.float64)
        self.w_err = torch.zeros(len(input.w))
        self.jacobian = BASparseMat(len(input.cams), len(input.x), self.p)

    def output(self):
        """Returns calculation result."""

        return BAOutput(
            self.reproj_error.detach().numpy(),
            self.w_err.detach().numpy(),
            self.jacobian,
        )

    def calculate_objective(self, times):
        """Calculates objective function many times."""

        reproj_error = torch.empty((self.p, 2), dtype=torch.float64)
        for i in range(times):
            for j in range(self.p):
                reproj_error[j] = compute_reproj_err(
                    self.cams[self.obs[j, 0]],
                    self.x[self.obs[j, 1]],
                    self.w[j],
                    self.feats[j],
                )

                self.w_err[j] = compute_w_err(self.w[j])

            self.reproj_error = reproj_error.flatten()

    def calculate_jacobian(self, times):
        """Calculates objective function jacobian many times."""

        reproj_error = torch.empty((self.p, 2), dtype=torch.float64)
        for i in range(times):
            # reprojection error jacobian calculation
            for j in range(self.p):
                camIdx = self.obs[j, 0]
                ptIdx = self.obs[j, 1]

                cam = self.cams[camIdx]
                x = self.x[ptIdx]
                w = self.w[j]

                reproj_error[j], J = torch_jacobian(
                    compute_reproj_err, (cam, x, w), (self.feats[j],)
                )

                self.jacobian.insert_reproj_err_block(j, camIdx, ptIdx, J)

            # weight error jacobian calculation
            for j in range(self.p):
                self.w_err[j], J = torch_jacobian(compute_w_err, (self.w[j],))

                self.jacobian.insert_w_err_block(j, J)

            self.reproj_error = reproj_error.flatten()


def calculate_jacobianBA(inputs):
    input = BAInput(
        inputs["cams"],
        inputs["X"],
        inputs["w"],
        inputs["obs"],
        inputs["feats"],
    )
    py = PyTorchBA()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.jacobian


def calculate_objectiveBA(inputs):
    input = BAInput(
        inputs["cams"],
        inputs["X"],
        inputs["w"],
        inputs["obs"],
        inputs["feats"],
    )
    py = PyTorchBA()
    py.prepare(input)
    py.calculate_objective(1)
    return (py.reproj_error, py.w_err)
