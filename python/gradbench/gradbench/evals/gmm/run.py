import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.gmm as golden
from gradbench.comparison import compare_json_objects
from gradbench.eval import SingleModuleValidatedEval, mismatch
from gradbench.evals.gmm import data_gen
from gradbench.wrap import Wrapped


def check(function: str, input: Any, output: Any) -> None:
    func: Wrapped = getattr(golden, function)
    expected = func.wrapped(input | {"runs": 1})["output"]
    return compare_json_objects(expected, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument(
        "-k", nargs="+", type=int, default=[5, 10, 25, 50]
    )  # misses 100 200
    parser.add_argument(
        "-d", nargs="+", type=int, default=[2, 10, 20, 32]
    )  # misses 64 128
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()
    e = SingleModuleValidatedEval(
        module="gmm", validator=None if args.no_validation else mismatch(check)
    )
    e.start()
    if e.define().success:
        n = args.n
        for d in args.d:
            for k in args.k:
                input = data_gen.main(d, k, n)
                e.evaluate(
                    function="objective",
                    input=input | {"runs": args.runs},
                    description=f"{d}_{k}_{n}",
                )
                e.evaluate(
                    function="jacobian",
                    input=input | {"runs": args.runs},
                    description=f"{d}_{k}_{n}",
                )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
