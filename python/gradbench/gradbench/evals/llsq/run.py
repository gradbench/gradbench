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
        module="llsq",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8196, 16392],
    )
    parser.add_argument("-m", nargs="+", type=int, default=[128])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="llsq", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        combinations = sorted(
            [(n, m) for n in args.n for m in args.m],
            key=lambda v: v[0] * v[1],
        )
        for n, m in combinations:
            x = np.random.rand(m).tolist()
            input = {"x": x, "n": n}
            e.evaluate(
                function="primal",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n},m={m}",
            )
            e.evaluate(
                function="gradient",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"n={n},m={m}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
