import argparse
import os.path
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.ht as golden
from gradbench.comparison import compare_json_objects
from gradbench.eval import SingleModuleValidatedEval, approve, mismatch
from gradbench.evals.ht import io
from gradbench.wrap import Wrapped


def check(function: str, input: Any, output: Any) -> None:
    func: Wrapped = getattr(golden, function)
    expected = func.wrapped(input | {"runs": 1})["output"]
    return compare_json_objects(expected, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(
        module="ht", validator=approve if args.no_validation else mismatch(check)
    )
    e.start()
    if e.define().success:
        data_root = Path("evals/ht/data")  # assumes cwd is set correctly
        # NOTE: data files are taken directly from ADBench.
        simple_small = data_root / "simple_small"
        simple_big = data_root / "simple_big"
        complicated_small = data_root / "complicated_small"
        complicated_big = data_root / "complicated_big"

        def evals(data_dir, complicated):
            c = os.path.basename(os.path.normpath(data_dir)) + "/"
            for i in range(args.min, args.max + 1):
                fn = next(data_dir.glob(f"hand{i}_*.txt"), None)
                model_dir = data_dir / "model"
                input = io.read_hand_instance(model_dir, fn, complicated).to_dict()
                e.evaluate(
                    function="objective",
                    input=input | {"runs": args.runs},
                    description=c + fn.stem,
                )
                e.evaluate(
                    function="jacobian",
                    input=input | {"runs": args.runs},
                    description=c + fn.stem,
                )

        evals(simple_small, False)
        evals(simple_big, False)
        evals(complicated_small, True)
        evals(complicated_big, True)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
