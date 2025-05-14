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

# Changes: Incorporated changes from
# https://github.com/microsoft/ADBench/pull/211

import math

import torch


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
    return -out + k * (C - torch.special.multigammaln(0.5 * n, p))


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)

    to_from_idx = torch.nn.functional.pad(
        torch.cumsum(torch.arange(d - 1, 0, -1), 0) + d, (1, 0), value=d
    ) - torch.arange(1, d + 1)
    idx = torch.tril(torch.arange(d).expand((d, d)).T + to_from_idx[None, :], -1)
    Ls = icf[:, idx] * (idx > 0)[None, ...]

    xcentered = x[:, None, :] - means[None, ...]

    Lxcentered = Qdiags * xcentered + torch.einsum("ijk,mik->mij", Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered**2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = torch.logsumexp(inner_term, 1)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return (
        CONSTANT
        + slse
        - n * torch.logsumexp(alphas, 0)
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)
    )
