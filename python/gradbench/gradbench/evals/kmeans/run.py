import argparse
from typing import Any

import numpy as np
from gradbench import cpp
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    mismatch,
)


def expect(function: str, input: Any) -> EvaluateResponse:
    return cpp.evaluate(
        tool="manual",
        module="kmeans",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("-n", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("-d", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="kmeans", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        combinations = sorted(
            [(k, n, d) for k in args.k for n in args.n for d in args.d],
            key=lambda v: v[0] * v[1] * v[2],
        )
        for k, n, d in combinations:
            points = np.random.rand(n, d).tolist()
            # This guarantees each cluster is nonempty.
            centroids = points[-k:]
            input = {"points": points, "centroids": centroids}
            e.evaluate(
                function="cost",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"k={k},n={n},d={d}",
            )
            e.evaluate(
                function="dir",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"k={k},n={n},d={d}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
