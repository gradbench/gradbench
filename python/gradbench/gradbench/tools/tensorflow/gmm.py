import tensorflow as tf
from gradbench import wrap
from gradbench.tools.tensorflow.gmm_objective import gmm_objective


@tf.function(jit_compile=True)
def gmm_value(fixed, independent):
    return gmm_objective(**fixed, **independent)


@tf.function(jit_compile=True)
def gmm_gradient(fixed, independent):
    with tf.GradientTape() as tape:
        tape.watch(independent)
        obj = gmm_objective(**fixed, **independent)
    return tape.gradient(obj, independent)


def prepare(input):
    def tensor(value):
        return tf.constant(value, dtype=tf.float64)

    fixed = {
        "d": input["d"],
        "k": input["k"],
        "n": input["n"],
        "x": tensor(input["x"]),
        "m": input["m"],
        "gamma": input["gamma"],
    }
    independent = {name: tensor(input[name]) for name in ["alpha", "mu", "q", "l"]}
    return fixed, independent


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    return gmm_value(*input)


@wrap.multiple_runs(
    pre=prepare,
    post=lambda grad: {name: g.numpy().tolist() for name, g in grad.items()},
)
def jacobian(input):
    return gmm_gradient(*input)
