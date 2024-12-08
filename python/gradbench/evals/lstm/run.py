import argparse
from pathlib import Path
from typing import Any

import numpy as np

import gradbench.pytorch.lstm as golden
from gradbench.evals.lstm import io
from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion
from gradbench.wrap_module import Functions


def check(function: str, input: Any, output: Any) -> None:
    func: Functions = getattr(golden, function)
    expected = func.unwrap(func(func.prepare(input)))
    assert np.all(np.isclose(expected, output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=int, default=[2, 4])
    parser.add_argument("-c", nargs="+", type=int, default=[1024, 4096])
    args = parser.parse_args()

    e = SingleModuleValidatedEvaluation(module="lstm", validator=assertion(check))
    e.start()
    if e.define().success:
        data_root = Path("evals/lstm/data")  # assumes cwd is set correctly

        for l in args.l:
            for c in args.c:
                fn = next(data_root.glob(f"lstm_l{l}_c{1024}.txt"), None)
                input = io.read_lstm_instance(fn).to_dict()
                e.evaluate(
                    function="calculate_objectiveLSTM", input=input, description=fn.stem
                )
                e.evaluate(
                    function="calculate_jacobianLSTM", input=input, description=fn.stem
                )


if __name__ == "__main__":
    main()
