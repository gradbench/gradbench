#!/usr/bin/env python3

import numpy as np
import os
import json
import sys
import time
import futhark_data
import futhark_server

binaries = { 'gmm': os.path.join(os.path.dirname(__file__), 'gmm')
            }


def run(server, params):
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
        server.cmd_free('output0')
        server.cmd_free('output1')
        server.cmd_free('output2')
    server.cmd_free('alpha')
    server.cmd_free('means')
    server.cmd_free('icf')
    server.cmd_free('x')
    server.cmd_free('gamma')
    server.cmd_free('m')
    return {"output": str(output), "nanoseconds": {"evaluate": end - start}}

def main():
    server = None
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(server, message)
        elif message["kind"] == "define":
            if message["module"] in binaries:
                server = futhark_server.Server(binaries[message["module"]])
                response["success"] = True
            else:
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
