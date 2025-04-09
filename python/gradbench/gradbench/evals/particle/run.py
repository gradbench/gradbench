import argparse
from typing import Any

from gradbench.eval import EvaluateResponse, SingleModuleValidatedEval, mismatch


def expect(function: str, input: Any) -> EvaluateResponse:
    return {"success": True, "output": 0.2071918746486116}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(module="particle", validator=mismatch(expect))
    e.start()
    if e.define().success:
        input = {"w": 0.0}
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
