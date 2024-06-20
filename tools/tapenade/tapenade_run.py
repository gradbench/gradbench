import json
import sys
import time
from importlib import import_module

def resolve(name):
    functions = import_module("run_shells")
    return getattr(functions, name)


def run(params):
    func = resolve(params["name"])
    vals = [1.0*arg["value"] for arg in params["arguments"]]
    
    start = time.perf_counter_ns()
    ret = func(*vals)
    end = time.perf_counter_ns()
    
    return {"return": ret.stdout, "nanoseconds": end - start}


def main():
    cfg = json.load(sys.stdin)
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()

