import futhark_server
import numpy as np


def prepare(server, input):
    server.put_value("input", np.float64(input))


def square(server):
    server.cmd_call("square", "output", "input")
    return server.get_value("output")


def double(server):
    server.cmd_call("double", "output", "input")
    return server.get_value("output")
