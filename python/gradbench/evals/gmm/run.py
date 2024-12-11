import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.gmm as golden
from gradbench.comparison import compare_json_objects
from gradbench.evals.gmm import data_gen
from gradbench.evaluation import SingleModuleValidatedEvaluation, mismatch
from gradbench.wrap_module import Functions


def check(function: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, function)
    expected, _ = func.unwrap(func(func.prepare(input | {"runs": 1})))
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
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="gmm", validator=mismatch(check))
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
    main()
