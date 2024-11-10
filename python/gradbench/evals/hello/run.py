from pathlib import Path
from typing import Any

from gradbench.evaluation import SingleModuleValidatedEvaluation, assertion


def check(name: str, input: Any, output: Any) -> None:
    match name:
        case "double":
            assert output == input * 2
        case "square":
            assert output == input * input


def main():
    e = SingleModuleValidatedEvaluation(module="hello", validator=assertion(check))
    if e.define().success:
        x = 1.0
        for _ in range(4):
            y = e.evaluate(name="square", input=x).output
            x = e.evaluate(name="double", input=y).output
    e.end()


if __name__ == "__main__":
    main()
