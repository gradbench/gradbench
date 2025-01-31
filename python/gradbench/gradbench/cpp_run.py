# Generic main function suitable for use by all tools that make use of
# the command line API provided by cpp/adbench/main.h.

import json
import sys
import time
from importlib import import_module


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["function"])
    vals = params["input"]
    proc = func(vals)
    if proc.returncode == 0:
        ls = proc.stdout.splitlines()
        output = json.loads(ls[0])
        timings = list(map(json.loads, ls[1:]))
        return {"output": output, "timings": timings}
    else:
        return {
            "output": None,
            "timings": [],
            "status": proc.returncode,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        }


def main():
    try:
        for line in sys.stdin:
            message = json.loads(line)
            response = {}
            if message["kind"] == "evaluate":
                response = run(message)
            elif message["kind"] == "define":
                try:
                    functions = import_module(message["module"])
                    func = getattr(functions, "compile")
                    func()  # compiles C code
                    response["success"] = True
                except ModuleNotFoundError:
                    response["success"] = False
                except Exception as e:
                    response["success"] = False
                    response["error"] = str(e)
            print(json.dumps({"id": message["id"]} | response), flush=True)
    except (EOFError, BrokenPipeError):
        pass
