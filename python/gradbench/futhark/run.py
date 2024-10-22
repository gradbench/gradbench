#!/usr/bin/env python3

import numpy as np
import os
import json
import sys
import time
import futhark_data
import futhark_server
import subprocess

def server_prog_source(prog):
    return os.path.join(os.path.dirname(__file__), prog + ".fut")

def server_prog(prog):
    return os.path.join(os.path.dirname(__file__), prog)

def run_hello(params):
    with futhark_server.Server(server_prog('hello')) as server:
        server.put_value("input", np.float64(params["input"]))
        start=time.time()
        server.cmd_call(params["name"], "output", "input")
        output = server.get_value('output')
        end=time.time()
        return {"output": str(output), "nanoseconds": {"evaluate": end - start}}

def run_gmm(params):
    with futhark_server.Server(server_prog('gmm')) as server:
        d = params["input"]["d"]
        k = params["input"]["k"]
        n = params["input"]["n"]
        alpha = np.array(params["input"]["alpha"], dtype=np.float64)
        means = np.array(params["input"]["means"], dtype=np.float64)
        icf = np.array(params["input"]["icf"], dtype=np.float64)
        x = np.array(params["input"]["x"], dtype=np.float64)
        gamma = np.float64(params["input"]["gamma"])
        m = np.int64(params["input"]["m"])
        server.put_value('alpha', alpha)
        server.put_value('means', means)
        server.put_value('icf', icf)
        server.put_value('x', x)
        server.put_value('gamma', gamma)
        server.put_value('m', m)
        if params["name"] == "calculate_objectiveGMM":
            start=time.time()
            server.cmd_call('calculate_objective',
                            'output',
                            'alpha',
                            'means',
                            'icf',
                            'x',
                            'gamma',
                            'm')
            end=time.time()
            output = server.get_value('output')
            server.cmd_free('output')
        else:
            start=time.time()
            server.cmd_call('calculate_jacobian',
                            'output0',
                            'output1',
                            'output2',
                            'alpha',
                            'means',
                            'icf',
                            'x',
                            'gamma',
                            'm')
            end=time.time()
            output = (server.get_value('output0'),
                      server.get_value('output1'),
                      server.get_value('output2'))
        return {"output": str(output), "nanoseconds": {"evaluate": end - start}}

modules = { 'hello': run_hello,
             'gmm': run_gmm
           }

FUTHARK_BACKEND='c'

def main():
    server = None
    run = None
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = modules[message["module"]](message)
        elif message["kind"] == "define":
            c = subprocess.run(['futhark',
                                FUTHARK_BACKEND,
                                '--server',
                                server_prog_source(message["module"])])
            response["success"] = c.returncode == 0
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
