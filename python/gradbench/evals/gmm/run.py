import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.gmm as golden
from gradbench.evals.gmm import data_gen
from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion
from gradbench.wrap_module import Functions


def check(function: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, function)
    expected = func.unwrap(func(func.prepare(input)))
    assert np.all(np.isclose(expected, output, rtol=1e-02))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("-k", nargs="+", type=int, default=[5, 10, 25, 50, 100, 200])
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="gmm", validator=assertion(check))
    e.start()
    if e.define().success:
        for n in args.n:
            for k in args.k:
                d = 2
                input = data_gen.main(d, k, n)
                e.evaluate(
                    function="calculate_objectiveGMM",
                    input=input,
                    description=f"{d}_{k}_{n}",
                )
                e.evaluate(
                    function="calculate_jacobianGMM",
                    input=input,
                    description=f"{d}_{k}_{n}",
                )


if __name__ == "__main__":
    main()
