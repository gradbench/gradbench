import numpy as np
import os
import futhark_server

def prepare(server, params):
    server.put_value("input", np.float64(params["input"]))

def square(server):
    server.cmd_call("square", "output", "input")
    return server.get_value('output')

def double(server):
    server.cmd_call("double", "output", "input")
    return server.get_value('output')
