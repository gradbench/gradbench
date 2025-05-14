import argparse
import traceback
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


def multivariate_gaussian_pdf(x, *, k, mu, Sigma_inverse):
    return np.exp(-(1 / 2) * (x - mu).T @ Sigma_inverse @ (x - mu)) / np.sqrt(
        (2 * np.pi) ** k / np.linalg.det(Sigma_inverse)
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


def log_posterior(*, D, K, N, x, m, gamma, mu, q, l, alpha):
    exp_alpha = np.exp(alpha)
    phi = exp_alpha / np.sum(exp_alpha)

    Q = []
    for k in range(K):
        Qk = np.zeros((D, D))
        for j in range(D):
            Qk[j, j] = np.exp(q[k][j])
            for i in range(j):
                Qk[j, i] = l[k][j][i]
        Q.append(Qk)
    Sigma_inverse = [Qk.T @ Qk for Qk in Q]

    likelihood = 1.0
    for i in range(N):
        likelihood_factor = 0.0
        for k in range(K):
            likelihood_factor += phi[k] * multivariate_gaussian_pdf(
                x[i], k=D, mu=mu[k], Sigma_inverse=Sigma_inverse[k]
            )
        likelihood *= likelihood_factor

    prior = 1.0
    for k in range(K):
        prior *= wishart_pdf(
            Sigma_inverse[k],
            p=D,
            n=D + m + 1,
            V=(1 / (gamma * gamma)) * np.identity(D),
        )

    return float(np.log(likelihood * prior))


def expect(function: str, input: Any) -> EvaluateResponse:
    return cpp.evaluate(
        tool="manual",
        module="gmm",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def expect_naive_objective(function: str, input: Any) -> EvaluateResponse:
    if function == "objective":
        try:
            D = input["d"]
            K = input["k"]
            icf = input["icf"]
            q = []
            l = []
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
        except Exception as e:
            return {"success": False, "error": "".join(traceback.format_exception(e))}
    else:
        return expect(function, input)


def generate(*, seed, d, k, n, m, gamma):
    rng = np.random.default_rng(seed=seed)
    return {
        "d": d,
        "k": k,
        "n": n,
        "x": [list(rng.normal(size=d)) for _ in range(n)],
        "m": m,
        "gamma": gamma,
        "mu": [list(rng.uniform(size=d)) for _ in range(k)],
        "q": [list(rng.normal(size=d)) for _ in range(k)],
        "l": [list(rng.normal(size=d * (d - 1) // 2)) for _ in range(k)],
        "alpha": list(rng.normal(size=k)),
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31337,
        help="seed for generating inputs",
    )
    parser.add_argument(
        "-d",
        nargs="+",
        type=int,
        default=[2, 10, 20, 32, 64],  # misses 128
        help="number of dimensions",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[5, 10, 25, 50, 100],  # misses 200
        help="number of mixture components",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        help="number of observations",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=0,
        help="additional degrees of freedom for Wishart distribution",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="precision for Wishart distribution",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="minimum number of times the tool should repeat each evaluation",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1,
        help="minimum seconds for which the tool should repeat each evaluation",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="do not validate",
    )
    parser.add_argument(
        "--validate-naive",
        action="store_true",
        help="use a naive implementation to validate the objective",
    )
    args = parser.parse_args()

    e = SingleModuleValidatedEval(
        module="gmm",
        validator=approve
        if args.no_validation
        else mismatch(expect_naive_objective if args.validate_naive else expect),
    )
    e.start(config=vars(args))
    if e.define().success:
        n = args.n
        combinations = sorted(
            [(d, k) for d in args.d for k in args.k],
            key=lambda v: v[0] * v[1],
        )
        for d, k in combinations:
            input = generate(seed=args.seed, d=d, k=k, n=n, m=args.m, gamma=args.gamma)
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
