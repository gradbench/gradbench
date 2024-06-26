# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np

from pytorch_defs import Wishart
from gmm_data import GMMInput

DELIM = ':'

def parse_floats(arr):
    '''Parses enumerable as float.

    Args:
        arr (enumerable): input data that can be parsed to floats.

    Returns:
        (List[float]): parsed data.
    '''
    
    return [ float(x) for x in arr ]



def read_gmm_instance(fn, replicate_point):
    '''Reads input data for GMM objective from the given file.

    Args:
        fn (str): input file name.
        replicate_point (bool): if False then file contains n different points,
            otherwise file contains only one point that will be replicated
            n times.
    
    Returns:
        (GMMInput): data for GMM objective test class.
    '''

    fid = open(fn, "r")

    line = fid.readline()
    line = line.split()

    d = int(line[0])
    k = int(line[1])
    n = int(line[2])

    alphas = np.array([ float(fid.readline()) for _ in range(k) ])
    means = np.array([ parse_floats(fid.readline().split()) for _ in range(k) ])
    icf = np.array([ parse_floats(fid.readline().split()) for _ in range(k) ])

    if replicate_point:
        x_ = parse_floats(fid.readline().split())
        x = np.array([ x_ ] * n)
    else:
        x = np.array([ parse_floats(fid.readline().split()) for _ in range(n) ])

    line = fid.readline().split()
    wishart_gamma = float(line[0])
    wishart_m = int(line[1])

    fid.close()

    return GMMInput(
        alphas,
        means,
        icf,
        x,
        Wishart(wishart_gamma, wishart_m)
    )



