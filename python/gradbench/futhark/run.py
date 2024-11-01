#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import time
from importlib import import_module

import futhark_server
import numpy as np


def server_prog_source(prog):
    return os.path.join(os.path.dirname(__file__), prog + ".fut")


def server_prog(prog):
    return os.path.join(os.path.dirname(__file__), prog)


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    with futhark_server.Server(server_prog(params["module"])) as server:
        prepare = resolve(params["module"], "prepare")
        run = resolve(params["module"], params["name"])
        prepare(server, params)
        start = time.perf_counter_ns()
        ret = run(server)
        end = time.perf_counter_ns()
        return {"output": ret, "nanoseconds": {"evaluate": end - start}}


FUTHARK_BACKEND = "c"


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            c = subprocess.run(
                [
                    "futhark",
                    FUTHARK_BACKEND,
                    "--server",
                    server_prog_source(message["module"]),
                ]
            )
            response["success"] = c.returncode == 0
        print(
            json.dumps({"id": message["id"], "tool": "futhark"} | response), flush=True
        )


if __name__ == "__main__":
    main()
