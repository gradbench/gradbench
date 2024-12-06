import json
import sys
import time
from importlib import import_module


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["name"])
    vals = params["input"]

    # start = time.perf_counter_ns()
    ret, time = func(vals).stdout.split("\n")
    # end = time.perf_counter_ns()

    return {"output": json.loads(ret), "nanoseconds": {"evaluate": int(time)}}


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
