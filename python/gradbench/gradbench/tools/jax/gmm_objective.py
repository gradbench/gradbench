import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, multigammaln


def log_wishart_prior(*, k, p, gamma, m, sum_qs, Qdiags, l):
    n = p + m + 1

    out = jnp.sum(
        0.5 * gamma * gamma * (jnp.sum(Qdiags**2, axis=1) + jnp.sum(l**2, axis=1))
        - m * sum_qs
    )

    C = n * p * (jnp.log(gamma / math.sqrt(2)))
    return -out + k * (C - multigammaln(0.5 * n, p))


def gmm_objective(*, d, k, n, x, m, gamma, alpha, mu, q, l):
    Qdiags = jnp.exp(q)
    sum_qs = jnp.sum(q, axis=1)

    row_idx, col_idx = jnp.tril_indices(d, -1)
    order = jnp.argsort(col_idx * d + row_idx)

    def make_l(l_row):
        return jnp.zeros((d, d)).at[row_idx[order], col_idx[order]].set(l_row)

    Ls = jax.vmap(make_l)(l)

    xcentered = x[:, None, :] - mu[None, ...]
    Lxcentered = Qdiags * xcentered + jnp.einsum("ijk,mik->mij", Ls, xcentered)
    sqsum_Lxcentered = jnp.sum(Lxcentered**2, axis=2)
    inner_term = alpha + sum_qs - 0.5 * sqsum_Lxcentered
    slse = jnp.sum(logsumexp(inner_term, axis=1))

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return (
        CONSTANT
        + slse
        - n * logsumexp(alpha, axis=0)
        + log_wishart_prior(
            k=k, p=d, gamma=gamma, m=m, sum_qs=sum_qs, Qdiags=Qdiags, l=l
        )
    )
