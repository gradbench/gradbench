import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.ba as golden
from gradbench.comparison import compare_json_objects
from gradbench.evaluation import SingleModuleValidatedEvaluation, mismatch
from gradbench.wrap_module import Functions


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


def check(function: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, function)
    expected, _ = func.unwrap(func(func.prepare(input | {"runs": 1})))
    return compare_json_objects(expected, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="ba", validator=mismatch(check))
    e.start()
    if e.define().success:
        # NOTE: data files are taken directly from ADBench. See README for more information.
        # Currently set to run on the smallest two data files. To run on all 20 set loop range to be: range(1,21)
        for i in range(args.min, args.max + 1):
            datafile = next((Path(__file__).parent / "data").glob(f"ba{i}_*.txt"), None)
            if datafile:
                input = parse(datafile)
                e.evaluate(
                    function="objective",
                    input=input | {"runs": args.runs},
                    description=datafile.stem,
                )
                e.evaluate(
                    function="jacobian",
                    input=input | {"runs": args.runs},
                    description=datafile.stem,
                )


if __name__ == "__main__":
    main()
