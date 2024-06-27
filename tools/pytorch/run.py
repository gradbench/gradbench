import json
import sys
import time
from importlib import import_module

import torch

# NOTE Works with JSON with proper input, currently have set to ignore input so doesn't mess with actions

def resolve(name):
    functions = import_module("GMM")
    return getattr(functions, "calculate_jacobian")

# NOTE Only want to call this when input is a number
def tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)

def run():
    func = resolve(name)
    # vals = [tensor(arg["value"]) for arg in params["arguments"]]
    vals = "d2_k5.txt"
    start = time.perf_counter_ns()
    # ret = func(*vals)
    ret = func(vals)
    end = time.perf_counter_ns()
    # return {"return": ret.item(), "nanoseconds": end - start}
    return {"return": ret.tolist(), "nanoseconds": end - start}


def main():
    # cfg = json.load(sys.stdin)
    # outputs = [run(params) for params in cfg["inputs"]]
    # print(json.dumps({"outputs": outputs}))
    print(json.dumps({"outputs": run()}))


if __name__ == "__main__":
    main()