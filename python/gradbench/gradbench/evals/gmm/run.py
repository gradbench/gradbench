import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gradbench.comparison import compare_json_objects
from gradbench.eval import SingleModuleValidatedEval, mismatch
from gradbench.evals.gmm import data_gen


def check(function: str, input: Any, output: Any, golden: Optional[Any]) -> None:
    if golden is not None:
        return compare_json_objects(golden["output"], output)
    else:
        return None


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

    e = SingleModuleValidatedEval(module="gmm", validator=mismatch(check))
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
