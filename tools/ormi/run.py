import json
import sys
import traceback
from importlib import import_module
from pathlib import Path

from gradbench.wrap import OutputAndTimings, Wrapped

###
###
###


def evaluate(message) -> OutputAndTimings:
    module = import_module(message["module"])
    name = message["function"]
    function = getattr(module, name)

    if not isinstance(function, Wrapped):
        raise TypeError(f"{message['module']}.{name} must be wrapped")

    return function.wrapped(message["input"])


###
###
###


def main() -> None:
    supported = set(
        Path(__file__).with_name("evals.txt").read_text(encoding="utf-8").splitlines()
    )
    for line in sys.stdin:
        message = json.loads(line)
        response = {}

        try:
            if message["kind"] == "start":
                response["tool"] = "ormi"
            elif (
                message["kind"] in ("define", "evaluate")
                and message["module"] not in supported
            ):
                response["success"] = False
            elif message["kind"] == "define":
                import_module(message["module"])
                response["success"] = True
            elif message["kind"] == "evaluate":
                response = evaluate(message)

        except Exception as error:
            response = {
                "success": False,
                "error": "".join(traceback.format_exception(error)),
            }

        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
