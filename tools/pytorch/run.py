import json
import sys
import time
from importlib import import_module

import torch

# Works with JSON with proper input, currently have set to ignore input so doesn't mess with actions

def resolve():
    functions = import_module("pytorchGMM")
    return getattr(functions, "calculate_jacobian")

def run():
    func = resolve()
    vals = "d2_k5.txt"
    start = time.perf_counter_ns()
    ret = func(vals)
    end = time.perf_counter_ns()
    return {"return": ret.tolist(), "nanoseconds": end - start}


def main():
    # cfg = json.load(sys.stdin)
    #outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": run()}))


if __name__ == "__main__":
    main()
