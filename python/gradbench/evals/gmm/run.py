import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.gmm as golden
from gradbench.evals.gmm import data_gen
from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion
from gradbench.wrap_module import Functions


def check(name: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, name)
    expected = func.unwrap(func(func.prepare(input)))
    assert np.all(np.isclose(expected, output, rtol=1e-02))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument(
        "-k", nargs="+", type=int, default=[5, 10, 25, 50]
    )  # misses 100 200
    parser.add_argument(
        "-d", nargs="+", type=int, default=[2, 10, 20, 32]
    )  # misses 64 128
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="gmm", validator=assertion(check))
    if e.define().success:
        n = args.n
        for d in args.d:
            for k in args.k:
                input = data_gen.main(d, k, n)
                e.evaluate(
                    name="calculate_objectiveGMM",
                    workload=f"{d}_{k}_{n}",
                    input=input,
                )
                e.evaluate(
                    name="calculate_jacobianGMM",
                    workload=f"{d}_{k}_{n}",
                    input=input,
                )
    e.end()


if __name__ == "__main__":
    main()
