import argparse
import json
from pathlib import Path
from typing import Any

import manual.lstm as golden
from gradbench.comparison import compare_json_objects
from gradbench.eval import Analysis, SingleModuleValidatedEval, approve, mismatch
from gradbench.evals.lstm import io


def check(function: str, input: Any, output: Any) -> None:
    func = getattr(golden, function)
    proc = func(input | {"min_runs": 1, "min_seconds": 0})
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
    parser.add_argument("-l", nargs="+", type=int, default=[2, 4])
    parser.add_argument("-c", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(
        module="lstm", validator=approve if args.no_validation else mismatch(check)
    )
    e.start(config=vars(args))
    if e.define().success:
        data_root = Path("evals/lstm/data")  # assumes cwd is set correctly

        for l in args.l:  # noqa: E741
            for c in args.c:
                fn = next(data_root.glob(f"lstm_l{l}_c{c}.txt"), None)
                input = io.read_lstm_instance(fn).to_dict()
                e.evaluate(
                    function="objective",
                    input=input
                    | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                    description=fn.stem,
                )
                e.evaluate(
                    function="jacobian",
                    input=input
                    | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                    description=fn.stem,
                )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
