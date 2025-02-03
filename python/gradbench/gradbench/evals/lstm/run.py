import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gradbench.comparison import compare_json_objects
from gradbench.eval import SingleModuleValidatedEval, mismatch
from gradbench.evals.lstm import io


def check(function: str, input: Any, output: Any, golden: Optional[Any]) -> None:
    if golden is not None:
        return compare_json_objects(golden["output"], output)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=int, default=[2, 4])
    parser.add_argument("-c", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="lstm", validator=mismatch(check))
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
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
