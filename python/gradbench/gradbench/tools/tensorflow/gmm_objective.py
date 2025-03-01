# Copyright (c) Microsoft Corporation.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowCommon/gmm_objective.py


import math

import tensorflow as tf
from scipy import special

from gradbench.tools.tensorflow.utils import shape


def logsumexp(x):
    mx = tf.reduce_max(x)
    return tf.reduce_logsumexp(x - mx) + mx


def logsumexpvec(x):
    """The same as "logsumexp" but calculates result for each row separately."""

    mx = tf.reduce_max(x, 1)
    lset = tf.reduce_logsumexp(tf.transpose(x) - mx, 0)
    return tf.transpose(lset + mx)


def log_gamma_distrib(a, p):
    return special.multigammaln(a, p)


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = shape(icf)[0]

    out = tf.reduce_sum(
        0.5
        * wishart_gamma
        * wishart_gamma
        * (tf.reduce_sum(Qdiags**2, 1) + tf.reduce_sum(icf[:, p:] ** 2, 1))
        - wishart_m * sum_qs
    )

    C = n * p * (math.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - log_gamma_distrib(0.5 * n, p))


def constructL(d, icf):
    constructL.Lparamidx = d

    def make_L_col(i):
        nelems = d - i - 1
        col = tf.concat(
            [
                tf.zeros(i + 1, dtype=tf.float64),
                icf[constructL.Lparamidx : (constructL.Lparamidx + nelems)],
            ],
            0,
        )

        constructL.Lparamidx += nelems
        return col

    columns = tuple(make_L_col(i) for i in range(d))
    return tf.stack(columns, -1)


def Qtimesx(Qdiag, L, x):
    return Qdiag * x + tf.linalg.matvec(L, x)


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    xshape = shape(x)
    n = xshape[0]
    d = xshape[1]

    Qdiags = tf.exp(icf[:, :d])
    sum_qs = tf.reduce_sum(icf[:, :d], 1)

    icf_sz = shape(icf)[0]
    Ls = tf.stack(tuple(constructL(d, icf[i]) for i in range(icf_sz)))

    xcentered = tf.stack(tuple(x[i] - means for i in range(n)))
    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    sqsum_Lxcentered = tf.reduce_sum(Lxcentered**2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    slse = tf.reduce_sum(lse)

    const = tf.constant(-n * d * 0.5 * math.log(2 * math.pi), dtype=tf.float64)

    return (
        const
        + slse
        - n * logsumexp(alphas)
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)
    )
