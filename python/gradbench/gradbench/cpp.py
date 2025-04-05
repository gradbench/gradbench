import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from typing import Any


def evaluate_completed_process(proc: subprocess.CompletedProcess[str]) -> Any:
    if proc.returncode == 0:
        ls = proc.stdout.splitlines()
        output = json.loads(ls[0])
        timings = list(map(json.loads, ls[1:]))
        return {"success": True, "output": output, "timings": timings}
    else:
        return {"success": False, "status": proc.returncode, "error": proc.stderr}


def define(*, args: argparse.Namespace, module: str) -> Any:
    try:
        subprocess.check_output(
            ["make", "-C", f"tools/{args.tool}", module, "-B", f"MULTITHREADED={'yes' if args.multithreaded else 'no'}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.output}
    except Exception as e:
        return {"success": False, "error": "".join(traceback.format_exception(e))}
    else:
        return {"success": True}


def evaluate(*, tool: str, module: str, function: str, input: Any) -> Any:
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return evaluate_completed_process(
            subprocess.run(
                [f"tools/{tool}/{module}", tmp.name, function],
                text=True,
                capture_output=True,
            )
        )


def run(args: argparse.Namespace) -> None:
    for line in sys.stdin:
        message = json.loads(line)
        response = {"id": message["id"]}
        match message["kind"]:
            case "start":
                response["tool"] = args.tool
            case "define":
                response |= define(args=args, module=message["module"])
            case "evaluate":
                response |= evaluate(
                    tool=args.tool,
                    module=message["module"],
                    function=message["function"],
                    input=message["input"],
                )
        print(json.dumps(response), flush=True)


def main() -> None:
    """
    Generic main function suitable for all tools that make use of the
    command line API provided by `cpp/adbench/main.hpp`.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    parser.add_argument("--multithreaded", action='store_true')
    args = parser.parse_args()

    try:
        run(args)
    except (EOFError, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
