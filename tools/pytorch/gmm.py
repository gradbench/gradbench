import torch


# shared
from utils import to_torch_tensors, torch_jacobian, to_torch_tensor
from itest import ITest
from defs import Wishart

# gmm
from gmm_data import GMMInput
from gmm_objective import gmm_objective

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
