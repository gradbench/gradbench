import argparse
from typing import Any

from gradbench import cpp
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    approve,
    mismatch,
)
from gradbench.evals.gmm import data_gen


def expect(function: str, input: Any) -> EvaluateResponse:
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
