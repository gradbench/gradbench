import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.ht as golden
from gradbench.evals.ht import io
from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion
from gradbench.wrap_module import Functions


def check(function: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, function)
    expected = func.unwrap(func(func.prepare(input)))
    assert np.all(np.isclose(expected, output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=2)
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="ht", validator=assertion(check))
    e.start()
    if e.define().success:
        data_root = Path("evals/ht/data")  # assumes cwd is set correctly
        # NOTE: data files are taken directly from ADBench.
        simple_small = data_root / "simple_small"
        simple_big = data_root / "simple_big"
        complicated_small = data_root / "complicated_small"
        complicated_big = data_root / "complicated_big"

        def evals(data_dir, complicated):
            # Shrink the range because some of the larger datasets take an
            # excessive amount of time with PyTorch.
            for i in range(args.min, args.max + 1):
                fn = next(data_dir.glob(f"hand{i}_*.txt"), None)
                model_dir = data_dir / "model"
                input = io.read_hand_instance(model_dir, fn, complicated).to_dict()
                e.evaluate(function="objective", input=input, description=fn.stem)
                e.evaluate(function="jacobian", input=input, description=fn.stem)

        evals(simple_small, False)
        evals(simple_big, False)
        evals(complicated_small, True)
        evals(complicated_big, True)


if __name__ == "__main__":
    main()
