import json
import sys
import time
from importlib import import_module

import tensorflow as tf


def resolve(): #name
    functions = import_module("GMM")
    # return getattr(functions, "name")
    return getattr(functions, "calculate_jacobian")

def tensor(x):
    return tf.Variable(x, dtype=tf.float64)

def run(): #params
    func = resolve()
    # vals = [tensor(arg["value"]) for arg in params["arguments"]]        
    vals = "d2_k5.txt"    
    start = time.perf_counter_ns()
    # ret = func(*vals)
    ret = func(vals)
    end = time.perf_counter_ns()
    # return {"return": ret.numpy(), "nanoseconds": end - start}
    return {"return": list(ret.numpy()), "nanoseconds": end - start}


def main():
    # cfg = json.load(sys.stdin)
    # outputs = [run(params) for params in cfg["inputs"]]
    # print(json.dumps({"outputs": outputs}))
    print(json.dumps({"outputs": run()}))


if __name__ == "__main__":
    main()
