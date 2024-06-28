import json
import sys
import time
from importlib import import_module

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
    vals = [tensor(arg["value"]) for arg in params["arguments"]]
    start = time.perf_counter_ns()
    ret = func(*vals)
    end = time.perf_counter_ns()
    return {"return": output(ret), "nanoseconds": end - start}



def main():
    cfg = json.load(sys.stdin)
    cfg["inputs"].append({'arguments': [{'value': 'd2_k5.txt'}], 'name': 'calculate_jacobianGMM'})
    cfg["inputs"].append({'arguments': [{'value': 'ba1_n49_m7776_p31843.txt'}], 'name': 'calculate_jacobianBA'})
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))

# {'arguments': [{'value': 'd2_k5.txt'}], 'name': 'calculate_jacobianGMM'}
# {'arguments': [{'value': 'ba1_n49_m7776_p31843.txt'}], 'name': 'calculate_jacobianBA'}

if __name__ == "__main__":
    main()