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

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/PyTorch/gmm_objective.py

import math

import torch
from scipy import special as scipy_special


def logsumexp(x):
    mx = torch.max(x)
    emx = torch.exp(x - mx)
    return torch.log(sum(emx)) + mx


def logsumexpvec(x):
    """The same as "logsumexp" but calculates result for each row separately."""

    mx = torch.max(x, 1).values
    lset = torch.logsumexp(torch.t(x) - mx, 0)
    return torch.t(lset + mx)


def log_gamma_distrib(a, p):
    return scipy_special.multigammaln(a, p)


def sqsum(x):
    return sum(x**2)


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]

    out = torch.sum(
        0.5
        * wishart_gamma
        * wishart_gamma
        * (torch.sum(Qdiags**2, dim=1) + torch.sum(icf[:, p:] ** 2, dim=1))
        - wishart_m * sum_qs
    )

    C = n * p * (math.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - log_gamma_distrib(0.5 * n, p))


def constructL(d, icf):
    constructL.Lparamidx = d

    def make_L_col(i):
        nelems = d - i - 1
        col = torch.cat(
            [
                torch.zeros(i + 1, dtype=torch.float64),
                icf[constructL.Lparamidx : (constructL.Lparamidx + nelems)],
            ]
        )

        constructL.Lparamidx += nelems
        return col

    columns = [make_L_col(i) for i in range(d)]
    return torch.stack(columns, -1)


def Qtimesx(Qdiag, L, x):

    f = torch.einsum("ijk,mik->mij", L, x)
    return Qdiag * x + f


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)
    Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])

    xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered**2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return (
        CONSTANT
        + slse
        - n * logsumexp(alphas)
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)
    )
