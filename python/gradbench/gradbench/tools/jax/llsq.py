import sys
import jax
import jax.numpy as jnp
import numpy as np
from gradbench import wrap
from functools import partial

jax.config.update("jax_enable_x64", True)


def t(i, n):
    return -1.0 + (i * 2.0) / (n - 1.0)


@partial(jax.jit, static_argnums=[1])
def llsq(x, n):
    m = x.shape[0]

    def f(i):
        ti = t(i, n)
        muls = jnp.full(m, ti).at[0].set(1)
        muls = jnp.cumprod(muls)
        g = lambda mul, xj: -(xj * mul)
        g_vals = jnp.vectorize(g)(muls, x)
        return (jnp.sign(ti) + jnp.sum(g_vals)) ** 2

    i_vals = jnp.arange(n)
    results = jax.vmap(f)(i_vals)

    return jnp.sum(results) / 2.0


@partial(jax.jit, static_argnums=[1])
def grad_llsq(x, n):
    return jax.grad(llsq, argnums=0)(x, n)


def prepare_input(input):
    return (np.array(input["x"], dtype=np.float64), input["n"])


@wrap.multiple_runs(pre=prepare_input, post=lambda x: float(x))
def primal(input):
    x, n = input
    return llsq(x, n)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def gradient(input):
    x, n = input
    return grad_llsq(x, n)
