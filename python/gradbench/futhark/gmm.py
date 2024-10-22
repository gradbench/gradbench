import numpy as np
import os
import futhark_server

def prepare(server, params):
    server.put_value('alpha', np.array(params["input"]["alpha"], dtype=np.float64))
    server.put_value('means', np.array(params["input"]["means"], dtype=np.float64))
    server.put_value('icf', np.array(params["input"]["icf"], dtype=np.float64))
    server.put_value('x', np.array(params["input"]["x"], dtype=np.float64))
    server.put_value('gamma', np.float64(params["input"]["gamma"]))
    server.put_value('m', np.int64(params["input"]["m"]))

def calculate_objectiveGMM(server):
    server.cmd_call('calculate_objective',
                    'output',
                    'alpha',
                    'means',
                    'icf',
                    'x',
                    'gamma',
                    'm')
    return server.get_value('output')

def calculate_jacobianGMM(server):
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
    return (server.get_value('output0'),
            server.get_value('output1'),
            server.get_value('output2'))
