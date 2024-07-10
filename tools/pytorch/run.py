import json
import sys
import time
from importlib import import_module
from pathlib import Path

import torch


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


# NOTE Only want to make tensor when input is a number
def tensor(x):
    if isinstance(x, (int, float, complex)):
        return torch.tensor(x, dtype=torch.float64, requires_grad=True)
    return x


def output(ret):
    if ret.size() == 1:
        return ret.item()
    return ret.tolist()


def run(params):
    func = resolve(params["module"], params["name"])
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
        elif message["kind"] == "define":
            try:
                import_module(message["module"])
                response["success"] = True
            except:
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
