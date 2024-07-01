import json
import sys
import time
from importlib import import_module


def resolve(name):
    functions = import_module("auto_functions")
    return getattr(functions, name)


def run(params):
    func = resolve(params["name"])
    arg = params["input"] * 1.0
    start = time.perf_counter_ns()
    ret = func(arg)
    end = time.perf_counter_ns()
    return {"output": ret, "nanoseconds": {"evaluate": end - start}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
