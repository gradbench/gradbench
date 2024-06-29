import json
import sys
import time
from importlib import import_module


def resolve(name):
    functions = import_module("run_shells")
    return getattr(functions, name)


def run(params):
    func = resolve(params["name"])
    vals = 1.0 * params["input"]

    # start = time.perf_counter_ns()
    ret, time = func(vals).stdout.split(" ")
    # end = time.perf_counter_ns()

    return {"output": float(ret), "nanoseconds": {"evaluate": int(time)}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
