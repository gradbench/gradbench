import argparse
import gzip
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.kmeans as golden
from gradbench.comparison import compare_json_objects
from gradbench.evals.lstm import io
from gradbench.evaluation import SingleModuleValidatedEvaluation, mismatch
from gradbench.wrap import Wrapped


def check(function: str, input: Any, output: Any) -> None:
    func: Wrapped = getattr(golden, function)
    expected = func.wrapped(input | {"runs": 1})["output"]
    return compare_json_objects(expected, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("-n", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("-d", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="kmeans", validator=mismatch(check))
    e.start()
    if e.define().success:
        for k in args.k:
            for n in args.n:
                for d in args.d:
                    input = {"k": k, "points": np.random.rand(n, d).tolist()}
                    e.evaluate(
                        function="kmeans",
                        input=input | {"runs": args.runs},
                        description=f"k={k},n={n},d={d}",
                    )


if __name__ == "__main__":
    main()
