import argparse
from pathlib import Path
from typing import Any

from gradbench import cpp
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    approve,
    mismatch,
)


def parse(file):
    lines = iter(Path(file).read_text().splitlines())

    n, m, p = [int(v) for v in next(lines).split()]

    one_cam = [float(x) for x in next(lines).split()]

    one_X = [float(x) for x in next(lines).split()]

    one_w = float(next(lines))

    one_feat = [float(x) for x in next(lines).split()]

    return {
        "n": n,
        "m": m,
        "p": p,
        "cam": one_cam,
        "x": one_X,
        "w": one_w,
        "feat": one_feat,
    }


def expect(function: str, input: Any) -> EvaluateResponse:
    return cpp.evaluate(
        tool="manual",
        module="ba",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=14)  # Can go up to 20
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(
        module="ba", validator=approve if args.no_validation else mismatch(expect)
    )
    e.start()
    if e.define().success:
        # NOTE: data files are taken directly from ADBench. See README for more information.
        for i in range(args.min, args.max + 1):
            datafile = next(Path("evals/ba/data").glob(f"ba{i}_*.txt"), None)
            if datafile:
                input = parse(datafile)
                e.evaluate(
                    function="objective",
                    input=input
                    | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                    description=datafile.stem,
                )
                e.evaluate(
                    function="jacobian",
                    input=input
                    | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                    description=datafile.stem,
                )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
