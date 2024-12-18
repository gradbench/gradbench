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
        output = ls[0]
        timings = [{"name": "evaluate", "nanoseconds": int(time)} for time in ls[1:]]
        return {"output": json.loads(output), "timings": timings}
    else:
        return {
            "output": None,
            "timings": [],
            "status": proc.returncode,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        }


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            try:
                functions = import_module(message["module"])
                func = getattr(functions, "compile")
                success = func()  # compiles C code
                response["success"] = success
            except:
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
