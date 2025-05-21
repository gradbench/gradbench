from functools import partial

import jax
import jax.numpy as jnp
from gradbench import wrap
from gradbench.tools.jax.gmm_objective import gmm_objective

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["d", "k"])
def obj(independent, **fixed):
    return gmm_objective(**fixed, **independent)


@partial(jax.jit, static_argnames=["d", "k"])
def jac(independent, **fixed):
    return jax.grad(lambda ind: gmm_objective(**fixed, **ind))(independent)


def prepare(input):
    def jaxify(*, d, k, n, x, m, gamma, alpha, mu, q, l, **_):
        return {
            "d": d,
            "k": k,
            "n": n,
            "x": jnp.array(x),
            "m": m,
            "gamma": gamma,
        }, {
            "alpha": jnp.array(alpha),
            "mu": jnp.array(mu),
            "q": jnp.array(q),
            "l": jnp.array(l),
        }

    return jaxify(**input)


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    fixed, independent = input
    return obj(independent, **fixed)


def postprocess(input):
    def listify(*, alpha, mu, q, l):
        return {
            "alpha": alpha.tolist(),
            "mu": mu.tolist(),
            "q": q.tolist(),
            "l": l.tolist(),
        }

    return listify(**input)


@wrap.multiple_runs(pre=prepare, post=postprocess)
def jacobian(input):
    fixed, independent = input
    return jac(independent, **fixed)
