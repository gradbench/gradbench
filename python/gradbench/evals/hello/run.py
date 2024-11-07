from pathlib import Path
from typing import Any

from gradbench.evaluation import SingleModuleValidatedEvaluation, correctness


def check(name: str, input: Any, output: Any) -> bool:
    match name:
        case "double":
            return output == input * 2
        case "square":
            return output == input * input


def main():
    e = SingleModuleValidatedEvaluation(module="hello", validator=correctness(check))
    if e.define(source=(Path(__file__).parent / "hello.adroit").read_text()).success:
        x = 1.0
        for _ in range(4):
            y = e.evaluate(name="square", input=x).output
            x = e.evaluate(name="double", input=y).output
    e.end()


if __name__ == "__main__":
    main()
