import json
import sys
import time
from importlib import import_module

# NOTE: The current implementation for Tapenade only supports passing in individual scalar inputs.


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["function"])
    vals = params["input"]
    ret, time = func(vals).stdout.split("\n")
    timings = [{"name": "evaluate", "nanoseconds": int(time)}]
    return {"output": json.loads(ret), "timings": timings}


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
