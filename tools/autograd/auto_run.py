import json
import sys
import time
from importlib import import_module


def resolve(name):
    functions = import_module("auto_functions")
    return getattr(functions, name)


def run(params):
    func = resolve(params["name"])
    vals = [arg["value"] * 1.0 for arg in params["arguments"]]
    start = time.perf_counter_ns()
    ret = func(*vals)
    end = time.perf_counter_ns()
    return {"return": ret, "nanoseconds": end - start}


def main():
    cfg = json.load(sys.stdin)
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()
