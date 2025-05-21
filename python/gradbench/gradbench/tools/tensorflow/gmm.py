import tensorflow as tf
from gradbench import wrap
from gradbench.tools.tensorflow.gmm_objective import gmm_objective


def prepare(input):
    def constant(value):
        return tf.constant(value, dtype=tf.float64)

    def variable(value):
        return tf.Variable(value, dtype=tf.float64)

    def tensify(*, d, k, n, x, m, gamma, alpha, mu, q, l, **_):
        return {
            "d": d,
            "k": k,
            "n": n,
            "x": constant(x),
            "m": m,
            "gamma": gamma,
            "alpha": variable(alpha),
            "mu": variable(mu),
            "q": variable(q),
            "l": variable(l),
        }

    return tensify(**input)


@wrap.multiple_runs(pre=prepare, post=float)
def objective(input):
    return gmm_objective(**input)


def postprocess(gradients):
    def listify(*, alpha, mu, q, l):
        return {
            "alpha": alpha.numpy().tolist(),
            "mu": mu.numpy().tolist(),
            "q": q.numpy().tolist(),
            "l": l.numpy().tolist(),
        }

    return listify(**gradients)


@wrap.multiple_runs(pre=prepare, post=postprocess)
def jacobian(input):
    def independent(*, alpha, mu, q, l, **_):
        return {"alpha": alpha, "mu": mu, "q": q, "l": l}

    with tf.GradientTape() as tape:
        obj = gmm_objective(**input)
    return tape.gradient(obj, independent(**input))
