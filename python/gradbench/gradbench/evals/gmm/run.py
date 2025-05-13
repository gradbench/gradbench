import argparse
from typing import Any

import numpy as np
import scipy
from gradbench import cpp
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    approve,
    mismatch,
)
from gradbench.evals.gmm import data_gen


def multivariate_normal_pdf(x, *, k, mu, Sigma):
    return np.exp(-(1 / 2) * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)) / np.sqrt(
        (2 * np.pi) ** k * np.linalg.det(Sigma)
    )


def wishart_pdf(X, *, p, n, V):
    return (
        np.linalg.det(X) ** ((n - p - 1) / 2)
        * np.exp(-np.trace(np.linalg.inv(V) @ X) / 2)
    ) / (
        2 ** ((p * n) / 2)
        * np.linalg.det(V) ** (n / 2)
        * np.exp(scipy.special.multigammaln(n / 2, p))
    )


def log_posterior(*, D, K, N, x, m, gamma, mu, q, l, alpha):  # noqa: E741
    exp_alpha = np.exp(alpha)
    sum_exp_alpha = np.sum(exp_alpha)
    phi = exp_alpha / sum_exp_alpha

    Q = []
    for k in range(K):
        Qk = np.zeros((D, D))
        for j in range(D):
            Qk[j, j] = np.exp(q[k][j])
            for i in range(j):
                Qk[j, i] = l[k][j][i]
        Q.append(Qk)
    Sigma_inverse = [Qk.T @ Qk for Qk in Q]
    Sigma = [np.linalg.inv(Sk) for Sk in Sigma_inverse]

    log_likelihood = 0.0
    for i in range(N):
        likelihood_factor = 0.0
        for k in range(K):
            likelihood_factor += phi[k] * multivariate_normal_pdf(
                x[i], k=D, mu=mu[k], Sigma=Sigma[k]
            )
        log_likelihood += np.log(likelihood_factor)

    log_prior = 0.0
    for k in range(K):
        log_prior += np.log(
            wishart_pdf(
                Sigma[k], p=D, n=D + m + 1, V=(1 / (gamma * gamma)) * np.identity(D)
            )
        )

    return float(log_likelihood + log_prior)


def expect(function: str, input: Any) -> EvaluateResponse:
    if function == "objective":
        D = input["d"]
        K = input["k"]
        icf = input["icf"]
        q = []
        l = []  # noqa: E741
        for k in range(K):
            i = D
            q.append(icf[k][:i])
            lk = [[] for _ in range(D)]
            for j in range(D):
                for row in range(j + 1, D):
                    lk[row].append(icf[k][i])
                    i += 1
            l.append(lk)
        return {
            "success": True,
            "output": log_posterior(
                D=D,
                K=K,
                N=input["n"],
                x=[np.array(xi) for xi in input["x"]],
                m=input["m"],
                gamma=input["gamma"],
                mu=[np.array(mui) for mui in input["means"]],
                q=q,
                l=l,
                alpha=input["alpha"],
            ),
        }
    return cpp.evaluate(
        tool="manual",
        module="gmm",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument(
        "-k", nargs="+", type=int, default=[5, 10, 25, 50, 100]
    )  # misses 200
    parser.add_argument(
        "-d", nargs="+", type=int, default=[2, 10, 20, 32, 64]
    )  # misses 128
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()
    e = SingleModuleValidatedEval(
        module="gmm", validator=approve if args.no_validation else mismatch(expect)
    )
    e.start(config=vars(args))
    if e.define().success:
        n = args.n
        combinations = sorted(
            [(d, k) for d in args.d for k in args.k],
            key=lambda v: v[0] * v[1],
        )
        for d, k in combinations:
            input = data_gen.main(d, k, n)
            e.evaluate(
                function="objective",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"{d}_{k}_{n}",
            )
            e.evaluate(
                function="jacobian",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"{d}_{k}_{n}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
