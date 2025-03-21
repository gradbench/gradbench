import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from typing import Any


def define(*, tool: str, module: str) -> Any:
    try:
        subprocess.check_output(
            ["make", "-C", f"tools/{tool}", f"run_{module}", "-B"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.output}
    else:
        return {"success": True}


def evaluate(*, tool: str, module: str, function: str, input: Any) -> Any:
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        proc = subprocess.run(
            [f"tools/{tool}/run_{module}", tmp.name, function],
            text=True,
            capture_output=True,
        )
    if proc.returncode == 0:
        ls = proc.stdout.splitlines()
        output = json.loads(ls[0])
        timings = list(map(json.loads, ls[1:]))
        return {"success": True, "output": output, "timings": timings}
    else:
        return {"success": False, "status": proc.returncode, "error": proc.stderr}


def run(tool: str) -> None:
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        match message["kind"]:
            case "start":
                response["tool"] = tool
            case "define":
                try:
                    response |= define(tool=tool, module=message["module"])
                except Exception as e:
                    response["success"] = False
                    response["error"] = "".join(traceback.format_exception(e))
            case "evaluate":
                response |= evaluate(
                    tool=tool,
                    module=message["module"],
                    function=message["function"],
                    input=message["input"],
                )
        print(json.dumps({"id": message["id"]} | response), flush=True)


def main() -> None:
    """
    Generic main function suitable for all tools that make use of the
    command line API provided by `cpp/adbench/main.hpp`.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    args = parser.parse_args()

    try:
        run(args.tool)
    except (EOFError, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
