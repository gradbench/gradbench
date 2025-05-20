import jax
import jax.numpy as jnp
import numpy as np
from gradbench import wrap
from jax import grad, jit

jax.config.update("jax_enable_x64", True)


def logsumexp(x):
    x_max = jnp.max(x)
    return jnp.log(jnp.sum(jnp.exp(x - x_max))) + x_max


def prepare_input(input):
    return np.array(input["x"], dtype=np.float64)


@wrap.multiple_runs(pre=prepare_input, post=float)
@jit
def primal(input):
    return logsumexp(input)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
@jit
def gradient(input):
    return grad(logsumexp)(input)
