import jax.numpy as jnp
from gradbench import wrap
from jax import grad


def to_tensor(x):
    return jnp.array(x, dtype=jnp.float32)


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def square(x):
    return x * x


@wrap.function(pre=to_tensor, post=lambda x: x.item())
def double(x):
    gradient = grad(square)
    return gradient(x)
