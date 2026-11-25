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
        module="ode",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    # After s RK4 steps, y[i] is identically zero for i >= 4*s, and RK4
    # approximates the analytic solution that the manual tool
    # differentiates only when h = 2/s is small compared to 1/n. These
    # defaults keep s >= 16*n, where they agree to about 1e-5.
    parser.add_argument("-n", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("-s", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="ode", validator=mismatch(expect))
    e.start()
    if e.define().success:
        np.random.seed(31337)  # For determinism.
        combinations = sorted(
            [(n, s) for n in args.n for s in args.s],
            key=lambda v: v[0] * v[1],
        )
        for n, s in combinations:
            # The exact y[n-1] at t=2 is prod(x) * 2**n / n!, so with x
            # in [0, 1) it underflows to zero for any n beyond a few
            # hundred. Scaling x[i] around (i+1)/2 keeps y[n-1], and
            # with it the gradient, of order one.
            x = ((np.arange(n) + 1) / 2 * np.random.uniform(0.9, 1.1, n)).tolist()
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
