import json
import sys
import time
from importlib import import_module

import jax.numpy as jnp
from jax import grad, jit


def resolve(name):
    functions = import_module("jax_functions")
    return getattr(functions, name)


def tensor(x):
    return jnp.array(x, dtype=jnp.float32)


def run(params):
    func = resolve(params["name"])
    vals = [tensor(arg["value"]) for arg in params["arguments"]]
    
    # jfunc = jit(func)
    
    start = time.perf_counter_ns()
    ret = func(*vals)
    end = time.perf_counter_ns()
    
    return {"return": ret.item(), "nanoseconds": end - start}


def main():
    cfg = json.load(sys.stdin)
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()
