#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from importlib import import_module

import futhark_server
import numpy as np


def server_prog_source(prog):
    return os.path.join("tools/futhark/", prog + ".fut")


def server_prog(prog):
    return os.path.join("tools/futhark/", prog)


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    with futhark_server.Server(server_prog(params["module"])) as server:
        prepare = resolve(params["module"], "prepare")
        run = resolve(params["module"], params["function"])
        prepare(server, params["input"])
        try:
            ret, times = run(server, params["input"])
            timings = [{"name": "evaluate", "nanoseconds": ns} for ns in times]
            return {"success": True, "output": ret, "timings": timings}
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="c")
    args = parser.parse_args()

    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            try:
                c = subprocess.check_output(
                    [
                        "futhark",
                        args.backend,
                        "--server",
                        server_prog_source(message["module"]),
                    ],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                response["success"] = False
                response["error"] = e.output
            else:
                response["success"] = True
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
