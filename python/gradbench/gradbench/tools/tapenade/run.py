import json
import sys
import argparse
from typing import Any

from gradbench import cpp
from gradbench.tools.tapenade import hello

tool = "tapenade"

args = argparse.Namespace()
args.tool = tool
args.multithreaded = False


def define(module: str) -> Any:
    match module:
        case "hello":
            success, error = hello.compile()
            response = {"success": success}
            if error is not None:
                response["error"] = error
            return response
        case _:
            return cpp.define(args=args, module=module)


def evaluate(*, module: str, function: str, input: Any) -> Any:
    match module:
        case "hello":
            return cpp.evaluate_completed_process(getattr(hello, function)(input))
        case _:
            return cpp.evaluate(
                tool=tool, module=module, function=function, input=input
            )


def run() -> None:
    for line in sys.stdin:
        message = json.loads(line)
        response = {"id": message["id"]}
        match message["kind"]:
            case "start":
                response["tool"] = tool
            case "define":
                response |= define(message["module"])
            case "evaluate":
                response |= evaluate(
                    module=message["module"],
                    function=message["function"],
                    input=message["input"],
                )
        print(json.dumps(response), flush=True)


def main() -> None:
    try:
        run()
    except (EOFError, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
