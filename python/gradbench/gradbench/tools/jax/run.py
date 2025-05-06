import argparse
import json
import os
import sys
import traceback
from importlib import import_module

import jax
from gradbench.wrap import Wrapped


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func: Wrapped = resolve(params["module"], params["function"])
    return func.wrapped(params["input"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multithreaded", action="store_true")
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    args = parser.parse_args()

    # It is very difficult to restrict Jax to a single thread, but we
    # can at least pin ourselves to the first CPU.
    if not args.multithreaded:
        os.sched_setaffinity(os.getpid(), [0])

    match args.precision:
        case "single":
            pass
        case "double":
            jax.config.update("jax_enable_x64", True)

    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "start":
            response["tool"] = "jax"
        elif message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            try:
                import_module(message["module"])
                response["success"] = True
            except Exception as e:
                response["error"] = "".join(traceback.format_exception(e))
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
