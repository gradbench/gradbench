from pathlib import Path
from typing import Any
import numpy as np

from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion


def check(name: str, input: Any, output: Any) -> None:
    match name:
        case "double":
            assert np.isclose(output, input * 2)
        case "square":
            assert np.isclose(output, input * input)


def main():
    e = SingleModuleValidatedEvaluation(module="hello", validator=assertion(check))
    if e.define().success:
        x = 1.0
        for _ in range(4):
            y = e.evaluate(name="square", workload=str(x), input=x).output
            x = e.evaluate(name="double", workload=str(x), input=y).output
    e.end()


if __name__ == "__main__":
    main()
