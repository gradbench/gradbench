from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.ht as golden
from gradbench.evals.ht import io
from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion
from gradbench.wrap_module import Functions


def check(name: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, name)
    expected = func.unwrap(func(func.prepare(input)))
    assert np.all(np.isclose(expected, output))


def main():
    e = SingleModuleValidatedEvaluation(module="ht", validator=assertion(check))
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
            for i in range(1, 13)[:2]:
                fn = next(data_dir.glob(f"hand{i}_*.txt"), None)
                model_dir = data_dir / "model"
                input = io.read_hand_instance(model_dir, fn, complicated).to_dict()
                e.evaluate(name="calculate_objectiveHT", input=input)
                e.evaluate(name="calculate_jacobianHT", input=input)

        evals(simple_small, False)
        evals(simple_big, False)
        evals(complicated_small, True)
        evals(complicated_big, True)

    e.end()


if __name__ == "__main__":
    main()
