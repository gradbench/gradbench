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

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Autograd/gmm_objective.py

import sys
import time as t

# import numpy as np
import autograd.numpy as np
from scipy import special as scipy_special


def logsumexp(x):
    # We need to use these element wise functions
    # because standard array level functions
    # did not work with Autograd
    def scalar_subtract_and_exp(a, scalar):
        return np.asarray([np.exp(a[i] - scalar) for i in range(a.size)])

    mx = np.amax(x)
    emx = scalar_subtract_and_exp(x, mx)
    return np.log(emx.sum()) + mx


def log_gamma_distrib(a, p):
    return scipy_special.multigammaln(a, p)


def sqsum(x):
    return (np.square(x)).sum()


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]
    out = 0
    for ik in range(k):
        frobenius = sqsum(Qdiags[ik, :]) + sqsum(icf[ik, p:])
        out = (
            out
            + 0.5 * wishart_gamma * wishart_gamma * frobenius
            - wishart_m * sum_qs[ik]
        )
    C = n * p * (np.log(wishart_gamma) - 0.5 * np.log(2)) - log_gamma_distrib(
        0.5 * n, p
    )
    return out - k * C


def constructL(d, icf):
    # Autograd does not support indexed assignment to arrays A[0,0] = x
    constructL.Lparamidx = d

    def make_L_col(i):
        nelems = d - i - 1
        col = np.concatenate(
            (
                np.zeros(i + 1),
                icf[constructL.Lparamidx : (constructL.Lparamidx + nelems)],
            )
        )
        constructL.Lparamidx += nelems
        return col

    columns = [make_L_col(i) for i in range(d)]
    return np.column_stack(columns)


def Qtimesx(Qdiag, L, x):
    # We need to use these element wise functions
    # because standard array level functions
    # did not work with Autograd
    def scalar_multiply(a, scalar):
        return np.asarray([(a[i] * scalar) for i in range(a.size)])

    def cwise_multiply(a, b):
        return np.asarray([(a[i] * b[i]) for i in range(a.size)])

    res = cwise_multiply(Qdiag, x)
    for i in range(L.shape[0]):
        res = res + scalar_multiply(L[:, i], x[i])
    return res


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    def inner_term(ix, ik):
        xcentered = x[ix, :] - means[ik, :]
        Lxcentered = Qtimesx(Qdiags[ik, :], Ls[ik, :, :], xcentered)
        sqsum_Lxcentered = sqsum(Lxcentered)
        return alphas[ik] + sum_qs[ik] - 0.5 * sqsum_Lxcentered

    n = x.shape[0]
    d = x.shape[1]
    k = alphas.size
    Qdiags = np.asarray([(np.exp(icf[ik, :d])) for ik in range(k)])
    sum_qs = np.asarray([(np.sum(icf[ik, :d])) for ik in range(k)])
    Ls = np.asarray([constructL(d, curr_icf) for curr_icf in icf])
    slse = 0
    for ix in range(n):
        lse = np.asarray([inner_term(ix, ik) for ik in range(k)])
        slse = slse + logsumexp(lse)

    CONSTANT = -n * d * 0.5 * np.log(2 * np.pi)
    return (
        CONSTANT
        + slse
        - n * logsumexp(alphas)
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)
    )
