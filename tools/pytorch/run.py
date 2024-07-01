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
    if str(x).isnumeric():
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
    return {"output": ret.item(), "nanoseconds": {"evaluate": end - start}}

def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        # if message["kind"] == "evaluate":
        response = run(message)
        print(json.dumps(response), flush=True)
        # print(json.dumps({"id": message["id"]} | response), flush=True)

# SAMPLE RUNS
# python ADBench_Data/GMM/gmm_data_parser.py ADBench_Data/GMM/d2_k5.txt |  docker run --interactive --rm "ghcr.io/gradbench/pytorch"
# echo '{"name": "double", "input": 3}' | docker run --interactive --rm ghcr.io/gradbench/pytorch
# echo -e '{ "name": "double", "input": 3} \n { "name": "double", "input": 3}' | docker run --interactive --rm ghcr.io/gradbench/pytorch


if __name__ == "__main__":
    main()
