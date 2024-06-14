import json
import sys
import time
from importlib import import_module

import tensorflow as tf


def resolve(name):
    functions = import_module("tensor_functions")
    return getattr(functions, name)


def tensor(x):
    return tf.Variable(x, dtype=tf.float64)


def run(params):
    func = resolve(params["name"])
    vals = [tensor(arg["value"]) for arg in params["arguments"]]        
    start = time.perf_counter_ns()
    ret = func(*vals)
    end = time.perf_counter_ns()

    return {"return": ret.numpy(), "nanoseconds": end - start}


def main():
    cfg = json.load(sys.stdin)
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()

