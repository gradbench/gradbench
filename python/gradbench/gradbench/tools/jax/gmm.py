import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from gradbench import wrap
from gradbench.adbench.defs import Wishart
from gradbench.adbench.gmm_data import GMMInput
from gradbench.tools.jax.gmm_objective import gmm_objective
from jax import grad, jit


@jit
def jax_jacobian(inputs, params):
    return grad(lambda inputs: gmm_objective(*inputs, *params))(inputs)


@jit
def jax_objective(inputs, params):
    return gmm_objective(*inputs, *params)


def prepare_input(input):
    return GMMInput(
        np.array(input["alpha"]),
        np.array(input["means"]),
        np.array(input["icf"]),
        np.array(input["x"]),
        Wishart(input["gamma"], input["m"]),
    )


@wrap.multiple_runs(
    pre=prepare_input,
    post=lambda x: x.tolist(),
)
def jacobian(inp):
    (a, b, c) = jax_jacobian(
        (inp.alphas, inp.means, inp.icf), (inp.x, inp.wishart.gamma, inp.wishart.m)
    )
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    return np.concatenate([a, b, c])


@wrap.multiple_runs(
    pre=prepare_input,
    post=lambda x: float(x),
)
def objective(inp):
    return jax_objective(
        (inp.alphas, inp.means, inp.icf), (inp.x, inp.wishart.gamma, inp.wishart.m)
    )
