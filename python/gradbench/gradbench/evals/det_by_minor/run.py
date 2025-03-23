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
        module="det_by_minor",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=int, default=[5, 6, 7, 8, 9, 10, 11])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="det_by_minor", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        for ell in args.l:
            A = np.random.rand(ell*ell).tolist()
            input = {"A": A, "ell": ell}
            e.evaluate(
                function="primal",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"{ell}",
            )
            e.evaluate(
                function="gradient",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"{ell}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
