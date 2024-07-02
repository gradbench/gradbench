import json
import sys
import time
from importlib import import_module

import numpy as np
from mygrad import tensor as mg_tensor


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def tensor(x):
    return mg_tensor(x, dtype=np.float64)


def run(params):
    func = resolve(params["module"],params["name"])
    vals = tensor(params["input"])
    start = time.perf_counter_ns()
    ret = func(vals)
    end = time.perf_counter_ns()
    return {"output": ret.item(), "nanoseconds": {"evaluate": end - start}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
            print(json.dumps({"id": message["id"]} | response), flush=True)
        elif message["kind"] == "define":
            try:
                import_module(message["module"])
                print(json.dumps({"id": message["id"], "success": True}), flush=True)
            except:
                print(json.dumps({"id": message["id"], "success": False}), flush=True)
        else:
             print(json.dumps({"id": message["id"]}), flush=True)

if __name__ == "__main__":
    main()
