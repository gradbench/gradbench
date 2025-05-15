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


def log_wishart_prior(*, k, p, gamma, m, sum_qs, Qdiags, l):
    n = p + m + 1

    out = torch.sum(
        0.5 * gamma * gamma * (torch.sum(Qdiags**2, dim=1) + torch.sum(l**2, dim=1))
        - m * sum_qs
    )

    C = n * p * (math.log(gamma / math.sqrt(2)))
    return -out + k * (C - torch.special.multigammaln(0.5 * n, p))


def gmm_objective(*, d, k, n, x, m, gamma, alpha, mu, q, l):
    Qdiags = torch.exp(q)
    sum_qs = torch.sum(q, 1)

    cols = torch.repeat_interleave(torch.arange(d - 1), torch.arange(d - 1, 0, -1))
    rows = torch.cat([torch.arange(c + 1, d) for c in range(d - 1)])
    Ls = torch.zeros((k, d, d), dtype=l.dtype, device=l.device)
    Ls[:, rows, cols] = l

    xcentered = x[:, None, :] - mu[None, ...]

    Lxcentered = Qdiags * xcentered + torch.einsum("ijk,mik->mij", Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered**2, 2)
    inner_term = alpha + sum_qs - 0.5 * sqsum_Lxcentered
    lse = torch.logsumexp(inner_term, 1)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return (
        CONSTANT
        + slse
        - n * torch.logsumexp(alpha, 0)
        + log_wishart_prior(
            k=k, p=d, gamma=gamma, m=m, sum_qs=sum_qs, Qdiags=Qdiags, l=l
        )
    )
