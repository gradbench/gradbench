import argparse
import json
from pathlib import Path
from typing import Any

import manual.gmm as golden
import numpy as np

from gradbench.comparison import compare_json_objects
from gradbench.eval import SingleModuleValidatedEval, approve, mismatch
from gradbench.evals.gmm import data_gen
from gradbench.wrap import Wrapped


def check(function: str, input: Any, output: Any) -> None:
    func = getattr(golden, function)
    proc = func(input | {"runs": 1})
    if proc.returncode == 0:
        ls = proc.stdout.splitlines()
        expected = json.loads(ls[0])
        return compare_json_objects(expected, output)
    else:
        return Analysis(
            valid=False,
            error=f"golden implementation failed with stderr:\n{proc.stderr}",
        )


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
        module="gmm", validator=approve if args.no_validation else mismatch(check)
    )
    e.start(config=vars(args))
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
