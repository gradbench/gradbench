import json
import sys
import time
from importlib import import_module
from pathlib import Path

import torch

def resolve(name):
    functions = import_module("functions")
    return getattr(functions, name)

# NOTE Only want to make tensor when input is a number
def tensor(x):
    if isinstance(x, (int, float, complex)):
        return torch.tensor(x, dtype=torch.float64, requires_grad=True)
    return x

def output(ret):
    if type(ret) == torch.Tensor:
        if ret.size() == 1: return ret.item()
        return ret.tolist()
    return ret.val_list()

def run(params):
    func = resolve(params["name"])
    arg = tensor(params["input"])
    start = time.perf_counter_ns()
    ret = func(arg)
    end = time.perf_counter_ns()
    return {"output": output(ret), "nanoseconds": {"evaluate": end - start}}

def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
