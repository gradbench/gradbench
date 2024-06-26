import json
import sys
import time
from importlib import import_module

import jax.numpy as jnp

# from jax import grad, jit


def resolve(name):
    functions = import_module("jax_functions")
    return getattr(functions, name)


def tensor(x):
    return jnp.array(x, dtype=jnp.float32)


def run(params):
    func = resolve(params["name"])
    arg = tensor(params["input"])

    # jfunc = jit(func)

    start = time.perf_counter_ns()
    ret = func(arg)
    end = time.perf_counter_ns()

    return {"output": ret.item(), "nanoseconds": {"evaluate": end - start}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
