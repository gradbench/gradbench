import argparse
from typing import Any

from gradbench.eval import EvaluateResponse, SingleModuleValidatedEval, mismatch


def expect(function: str, input: Any) -> EvaluateResponse:
    return {
        "success": True,
        "output": [
            8.246324826140356e-06,
            8.246324826140356e-06,
            8.246324826140356e-06,
            8.246324826140356e-06,
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="saddle", validator=mismatch(expect))
    e.start()
    if e.define().success:
        input = {"start": [1.0, 1.0]}
        e.evaluate(
            function="rr",
            input=input | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
            description="w=0",
        )
        e.evaluate(
            function="ff",
            input=input | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
            description="w=0",
        )
        e.evaluate(
            function="fr",
            input=input | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
            description="w=0",
        )
        e.evaluate(
            function="rf",
            input=input | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
            description="w=0",
        )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
