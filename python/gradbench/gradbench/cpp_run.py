# Generic main function suitable for use by all tools that make use of
# the command line API provided by cpp/adbench/main.h.

import json
import sys
import traceback
from importlib import import_module
from pathlib import Path


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
        return {"success": True, "output": output, "timings": timings}
    else:
        return {"success": False, "status": proc.returncode, "error": proc.stderr}


def main(pathname: str):
    tool = Path(pathname).parent.name
    try:
        for line in sys.stdin:
            message = json.loads(line)
            response = {}
            if message["kind"] == "start":
                response["tool"] = tool
            elif message["kind"] == "evaluate":
                response = run(message)
            elif message["kind"] == "define":
                try:
                    functions = import_module(message["module"])
                    func = getattr(functions, "compile")
                    success, error = func()
                    response["success"] = success
                    if not success:
                        response["error"] = error
                except ModuleNotFoundError:
                    response["success"] = False
                except Exception as e:
                    response["success"] = False
                    response["error"] = traceback.format_exc() + "\n" + str(e)
            print(json.dumps({"id": message["id"]} | response), flush=True)
    except (EOFError, BrokenPipeError):
        pass
