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
    inputs = tensor(params["input"])
    start = time.perf_counter_ns()
    ret = func(inputs )
    end = time.perf_counter_ns()
    return {"return": output(ret), "nanoseconds": end - start}

def main():
    for line in sys.stdin:
        cfg = json.loads(line)
        outputs = [run(cfg)]
        print(json.dumps({"outputs": outputs}))

# SAMPLE RUNS
# python ADBench_Data/GMM/gmm_data_parser.py ADBench_Data/GMM/d2_k5.txt |  docker run --interactive --rm "ghcr.io/gradbench/pytorch"
# echo '{"name": "double", "input": 3}' | docker run --interactive --rm ghcr.io/gradbench/pytorch
# echo -e '{ "name": "double", "input": 3} \n { "name": "double", "input": 3}' | docker run --interactive --rm ghcr.io/gradbench/pytorch


if __name__ == "__main__":
    main()