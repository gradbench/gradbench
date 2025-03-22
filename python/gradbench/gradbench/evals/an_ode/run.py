import argparse
import json
from typing import Any

import numpy as np
from gradbench import cpp
from gradbench.comparison import compare_json_objects
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    mismatch,
)


def expect(function: str, input: Any) -> EvaluateResponse:
    return cpp.evaluate(
        tool="manual",
        module="an_ode",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", nargs="+", type=int, default=[100000])
    parser.add_argument("-s", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="an_ode", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        combinations = sorted(
            [(n, s) for n in args.n for s in args.s],
            key=lambda v: v[0] * v[1],
        )
        for n, s in combinations:
            x = np.random.rand(n).tolist()
            input = {"x": x, "s": s}
            e.evaluate(
                function="primal",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n},s={s}",
            )
            e.evaluate(
                function="gradient",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n},s={s}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
