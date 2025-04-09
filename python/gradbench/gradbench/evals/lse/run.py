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
        module="lse",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", nargs="+", type=int, default=[10000, 100000, 1000000])
    parser.add_argument("--min", nargs="+", type=float, default=-100)
    parser.add_argument("--max", nargs="+", type=float, default=100)
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="lse", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        for n in args.n:
            x = np.random.rand(n).tolist()
            input = {"x": x}
            e.evaluate(
                function="primal",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n}",
            )
            e.evaluate(
                function="gradient",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
