import math
import jax.numpy as jnp
from jax.scipy.special import logsumexp, multigammaln


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]

    out = jnp.sum(
        0.5
        * wishart_gamma
        * wishart_gamma
        * (jnp.sum(Qdiags**2, axis=1) + jnp.sum(icf[:, p:] ** 2, axis=1))
        - wishart_m * sum_qs
    )

    C = n * p * (jnp.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - multigammaln(0.5 * n, p))


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n, d = x.shape

    Qdiags = jnp.exp(icf[:, :d])
    sum_qs = jnp.sum(icf[:, :d], axis=1)

    to_from_idx = jnp.pad(
        jnp.cumsum(jnp.arange(d - 1, 0, -1)) + d, (1, 0), constant_values=d
    ) - jnp.arange(1, d + 1)
    idx = jnp.tril(jnp.arange(d).reshape(d, 1) + to_from_idx, -1)
    Ls = icf[:, idx] * (idx > 0)

    xcentered = x[:, None, :] - means[None, ...]
    Lxcentered = Qdiags * xcentered + jnp.einsum("ijk,mik->mij", Ls, xcentered)
    sqsum_Lxcentered = jnp.sum(Lxcentered**2, axis=2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    slse = jnp.sum(logsumexp(inner_term, axis=1))

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return (
        CONSTANT
        + slse
        - n * logsumexp(alphas, axis=0)
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)
    )
