import numpy as np
import torch

# shared
from utils import to_torch_tensors, torch_jacobian, to_torch_tensor
from itest import ITest
from input_utils import read_gmm_instance, read_ba_instance
from defs import Wishart

# gmm
from gmm_data import GMMInput, GMMOutput
from gmm_objective import gmm_objective

# ba
from ba_data import BAInput, BAOutput
from ba_sparse_mat import BASparseMat
from ba_objective import compute_reproj_err, compute_w_err

def square(x):
    return x * x

def double(x):
    y = square(x)
    y.backward()
    return x.grad

class PyTorchGMM(ITest):
    '''Test class for GMM differentiation by PyTorch.'''

    def prepare(self, input):
        '''Prepares calculating. This function must be run before
        any others.'''

        self.inputs = to_torch_tensors(
            (input.alphas, input.means, input.icf),
            grad_req = True
        )

        self.params = to_torch_tensors(
            (input.x, input.wishart.gamma, input.wishart.m)
        )

        self.objective = torch.zeros(1)
        self.gradient = torch.empty(0)

    def output(self):
        '''Returns calculation result.'''

        return GMMOutput(self.objective.item(), self.gradient.numpy())

    def calculate_objective(self, times):
        '''Calculates objective function many times.'''

        for i in range(times):
            self.objective = gmm_objective(*self.inputs, *self.params)

    def calculate_jacobian(self, times):
        '''Calculates objective function jacobian many times.'''

        for i in range(times):
            self.objective, self.gradient = torch_jacobian(
                gmm_objective,
                self.inputs,
                self.params
            )

def calculate_jacobianGMM(inputs):
    input = GMMInput(
        inputs["alpha"],
        inputs["means"],
        inputs["icf"],
        inputs["x"],
        Wishart(inputs["gamma"], inputs["m"])
    )
    py = PyTorchGMM()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.gradient

def calculate_objectiveGMM(inputs):
    input = GMMInput(
        inputs["alpha"],
        inputs["means"],
        inputs["icf"],
        inputs["x"],
        Wishart(inputs["gamma"], inputs["m"])
    )
    py = PyTorchGMM()
    py.prepare(input)
    py.calculate_objective(1)
    return py.objective

class PyTorchBA(ITest):
    '''Test class for BA diferentiation by PyTorch.'''

    def prepare(self, input):
        '''Prepares calculating. This function must be run before
        any others.'''

        self.p = len(input.obs)

        # we use tuple of tensors instead of multidimensional tensor
        # because torch doesn't differentiate by non-leaf tensors
        self.cams = tuple(
            to_torch_tensor(cam, grad_req = True) for cam in input.cams
        )

        self.x = tuple(
            to_torch_tensor(x, grad_req = True) for x in input.x
        )

        self.w = tuple(
            to_torch_tensor(w, grad_req = True) for w in input.w
        )

        self.obs = to_torch_tensor(input.obs, dtype = torch.int64)
        self.feats = to_torch_tensor(input.feats)

        self.reproj_error = torch.zeros(2 * self.p, dtype = torch.float64)
        self.w_err = torch.zeros(len(input.w))
        self.jacobian = BASparseMat(len(input.cams), len(input.x), self.p)

    def output(self):
        '''Returns calculation result.'''

        return BAOutput(
            self.reproj_error.detach().numpy(),
            self.w_err.detach().numpy(),
            self.jacobian
        )

    def calculate_objective(self, times):
        '''Calculates objective function many times.'''

        reproj_error = torch.empty((self.p, 2), dtype = torch.float64)
        for i in range(times):
            for j in range(self.p):
                reproj_error[j] = compute_reproj_err(
                    self.cams[self.obs[j, 0]],
                    self.x[self.obs[j, 1]],
                    self.w[j],
                    self.feats[j]
                )

                self.w_err[j] = compute_w_err(self.w[j])

            self.reproj_error = reproj_error.flatten()

    def calculate_jacobian(self, times):
        ''' Calculates objective function jacobian many times.'''

        reproj_error = torch.empty((self.p, 2), dtype = torch.float64)
        for i in range(times):
            # reprojection error jacobian calculation
            for j in range(self.p):
                camIdx = self.obs[j, 0]
                ptIdx = self.obs[j, 1]

                cam = self.cams[camIdx]
                x = self.x[ptIdx]
                w = self.w[j]

                reproj_error[j], J = torch_jacobian(
                    compute_reproj_err,
                    ( cam, x, w ),
                    ( self.feats[j], )
                )

                self.jacobian.insert_reproj_err_block(j, camIdx, ptIdx, J)

            # weight error jacobian calculation
            for j in range(self.p):
                self.w_err[j], J = torch_jacobian(
                    compute_w_err,
                    ( self.w[j], )
                )

                self.jacobian.insert_w_err_block(j, J)

            self.reproj_error = reproj_error.flatten()

def calculate_jacobianBA(file):
    input = read_ba_instance(file)
    py = PyTorchBA()
    py.prepare(input)
    py.calculate_jacobian(1)
    return py.jacobian

def calculate_objectiveBA(file):
    input = read_ba_instance(file)
    py = PyTorchBA()
    py.prepare(input)
    py.calculate_objective(1)
    return py.reproj_error, py.w_err
