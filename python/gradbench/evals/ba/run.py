from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.ba as golden
from gradbench.evaluation import SingleModuleValidatedEvaluation, correctness
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


def check(name: str, input: Any, b: Any) -> bool:
    func: Functions = getattr(golden, name)
    a = func.unwrap(func(func.prepare(input)))
    match name:
        case "calculate_objectiveBA":
            return (
                np.all(
                    np.isclose(
                        a["reproj_error"]["elements"], b["reproj_error"]["elements"]
                    )
                )
                and a["reproj_error"]["repeated"] == b["reproj_error"]["repeated"]
                and np.all(np.isclose(a["w_err"]["element"], b["w_err"]["element"]))
                and a["w_err"]["repeated"] == b["w_err"]["repeated"]
            )
        case "calculate_jacobianBA":
            return a == b


def main():
    e = SingleModuleValidatedEvaluation(module="ba", validator=correctness(check))
    if e.define(source="PLACE HOLDER").success:
        # NOTE: data files are taken directly from ADBench. See README for more information.
        # Currently set to run on the smallest two data files. To run on all 20 set loop range to be: range(1,21)
        for i in range(1, 3):
            datafile = next((Path(__file__).parent / "data").glob(f"ba{i}_*.txt"), None)
            if datafile:
                input = parse(datafile)
                e.evaluate(name="calculate_objectiveBA", input=input)
                e.evaluate(name="calculate_jacobianBA", input=input)
    e.end()


if __name__ == "__main__":
    main()
