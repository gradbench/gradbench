from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.gmm as golden
from gradbench.evals.gmm import data_gen
from gradbench.evaluation import SingleModuleValidatedEvaluation, correctness
from gradbench.wrap_module import Functions


def check(name: str, input: Any, output: Any) -> bool:
    func: Functions = getattr(golden, name)
    expected = func.unwrap(func(func.prepare(input)))
    return np.all(np.isclose(expected, output))


def main():
    e = SingleModuleValidatedEvaluation(module="gmm", validator=correctness(check))
    if e.define(source=(Path(__file__).parent / "gmm.adroit").read_text()).success:
        for n in [1000, 10000]:
            for k in [5, 10, 25, 50, 100, 200]:
                input = data_gen.main(2, k, n)  # d k n
                e.evaluate(name="calculate_objectiveGMM", input=input)
                e.evaluate(name="calculate_jacobianGMM", input=input)
    e.end()


if __name__ == "__main__":
    main()
