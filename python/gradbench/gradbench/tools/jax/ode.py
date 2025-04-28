from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gradbench import wrap


# ODE function: depends on index i, x[i], and y[i-1] (rotated y)
def ode_fun_vec(x):
    def f(y):
        def single_component(i, xi, yi_rot):
            return jnp.where(i == 0, xi, xi * yi_rot)

        iota = jnp.arange(x.shape[0])
        y_rot = jnp.roll(y, shift=1)
        return jax.vmap(single_component)(iota, x, y_rot)

    return f


# Runge-Kutta 4th order ODE solver that returns the final state
def runge_kutta(f, yi, tf, s):
    h = tf / s

    def body_fun(yf, _):
        k1 = f(yf)
        k2 = f(yf + (h / 2 * k1))
        k3 = f(yf + h / 2 * k2)
        k4 = f(yf + (h * k3))
        yf_next = yf + (h / 6 * (k1 + ((2 * k2) + (2 * k3) + k4)))
        return yf_next, yf_next  # carry and collect output

    _, y_trajectory = jax.lax.scan(body_fun, yi, None, length=s)
    return y_trajectory[-1]


@partial(jax.jit, static_argnums=[1])
def ode(x, s):
    tf = 2.0
    f = ode_fun_vec(x)
    y0 = jnp.zeros_like(x)
    return runge_kutta(f, y0, tf, s)


@partial(jax.jit, static_argnums=[1])
def grad_ode(x, s):
    return jax.grad(lambda x_: ode(x_, s)[-1])(x)


def prepare_input(input):
    return (np.array(input["x"], dtype=np.float64), input["s"])


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def primal(input):
    x, s = input
    return ode(x, s)


@wrap.multiple_runs(pre=prepare_input, post=lambda x: x.tolist())
def gradient(input):
    x, s = input
    return grad_ode(x, s)
