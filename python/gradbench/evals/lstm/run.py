import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.lstm as golden
from gradbench.comparison import compare_json_objects
from gradbench.evals.lstm import io
from gradbench.evaluation import SingleModuleValidatedEvaluation, mismatch
from gradbench.wrap import Wrapped


def check(function: str, input: Any, output: Any) -> None:
    func: Wrapped = getattr(golden, function)
    expected = func.wrapped(input | {"runs": 1}).output
    return compare_json_objects(expected, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=int, default=[2, 4])
    parser.add_argument("-c", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="lstm", validator=mismatch(check))
    e.start()
    if e.define().success:
        data_root = Path("evals/lstm/data")  # assumes cwd is set correctly

        for l in args.l:
            for c in args.c:
                fn = next(data_root.glob(f"lstm_l{l}_c{c}.txt"), None)
                input = io.read_lstm_instance(fn).to_dict()
                e.evaluate(
                    function="objective",
                    input=input | {"runs": args.runs},
                    description=fn.stem,
                )
                e.evaluate(
                    function="jacobian",
                    input=input | {"runs": args.runs},
                    description=fn.stem,
                )


if __name__ == "__main__":
    main()
