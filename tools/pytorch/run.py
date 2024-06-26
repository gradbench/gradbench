import json
import sys
import time
from importlib import import_module

import torch


def resolve(name):
    functions = import_module("functions")
    return getattr(functions, name)


def tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def run(params):
    func = resolve(params["name"])
    arg = tensor(params["input"])
    start = time.perf_counter_ns()
    ret = func(arg)
    end = time.perf_counter_ns()
    return {"output": ret.item(), "nanoseconds": {"evaluate": end - start}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
