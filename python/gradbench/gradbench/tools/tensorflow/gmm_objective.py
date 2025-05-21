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


def log_wishart_prior(*, k, p, gamma, m, sum_qs, Qdiags, l):
    n = p + m + 1

    out = tf.reduce_sum(
        0.5 * gamma * gamma * (tf.reduce_sum(Qdiags**2, 1) + tf.reduce_sum(l**2, 1))
        - m * sum_qs
    )

    C = n * p * (math.log(gamma / math.sqrt(2)))
    return -out + k * (C - log_gamma_distrib(0.5 * n, p))


def constructL(*, d, l):
    j = 0

    def make_L_col(i):
        nonlocal j

        nelems = d - i - 1
        col = tf.concat([tf.zeros(i + 1, dtype=tf.float64), l[j : (j + nelems)]], 0)

        j += nelems
        return col

    columns = tuple(make_L_col(i) for i in range(d))
    return tf.stack(columns, -1)


def Qtimesx(Qdiag, L, x):
    return Qdiag * x + tf.linalg.matvec(L, x)


def gmm_objective(*, d, k, n, x, m, gamma, alpha, mu, q, l):
    Qdiags = tf.exp(q)
    sum_qs = tf.reduce_sum(q, 1)

    Ls = tf.stack(tuple(constructL(d=d, l=l[i]) for i in range(k)))

    xcentered = tf.stack(tuple(x[i] - mu for i in range(n)))
    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    sqsum_Lxcentered = tf.reduce_sum(Lxcentered**2, 2)
    inner_term = alpha + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    slse = tf.reduce_sum(lse)

    const = tf.constant(-n * d * 0.5 * math.log(2 * math.pi), dtype=tf.float64)

    return (
        const
        + slse
        - n * logsumexp(alpha)
        + log_wishart_prior(
            k=k, p=d, gamma=gamma, m=m, sum_qs=sum_qs, Qdiags=Qdiags, l=l
        )
    )
