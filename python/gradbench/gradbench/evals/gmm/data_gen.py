# Copyright (c) Microsoft Corporation.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm/gmm-data-gen.py

"""
Changes Made:
- main() now takes in 3 parameters for a specifc d,k, and n.
- generator() now takes in those specifc values rather than the arrays as a whole. It then calls generate once with those variables instead of using a for loop.
- Instead of writing to a file, generate() creates a dictionary with the new data. This dictionary is then returned from main()
- Commented out and removed support for 2.5M datapoints
- No longer obscures the origin of exceptions.
"""

import sys

import numpy as np


# function printing to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_points_dir_name(n):
    if n == 1000:
        return "1k"
    if n == 10000:
        return "10k"
    # if n == 2500000:
    #     return "2.5M"
    raise ValueError("Undefined number of points: {n}")


def replicate_point(n):
    if n == 1000:
        return False
    if n == 10000:
        return False
    # if n == 2500000:
    #     return True
    raise ValueError("Undefined number of points: {n}")


def generate(data_uniform, data_normal, D, k, n):
    gamma = 1.0
    m = 0

    view_uniform = data_uniform[:]
    view_normal = data_normal[:]

    filename = f"gmm_d{D}_K{k}_N{n}.txt"

    output = {}

    output["d"] = D
    output["k"] = k
    output["n"] = n

    # alpha
    output["alpha"] = []
    for i in range(k):
        output["alpha"].append(float(view_normal[0]))
        view_normal = view_normal[1:]

    # mu
    output["means"] = [[] for _ in range(k)]
    for i in range(k):
        for j in range(D):
            output["means"][i].append(float(view_uniform[0]))
            view_uniform = view_uniform[1:]

    # q
    output["icf"] = [[] for _ in range(k)]
    for i in range(k):
        for j in range(D + D * (D - 1) // 2):
            output["icf"][i].append(float(view_normal[0]))
            view_normal = view_normal[1:]

    # x
    output["x"] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(D):
            output["x"][i].append(float(view_normal[0]))
            view_normal = view_normal[1:]

    output["gamma"] = gamma
    output["m"] = m

    return output


def generator(d, k, n):
    np.random.seed(31337)  # For determinism.

    K_max = 200  # K[-1] from [5, 10, 25, 50, 100, 200]
    N_max = 10000  # N[-2] from [1000,10000,2500000]

    # uniform distribution parameters
    low = 0
    high = 1
    amount_of_uniform_numbers = K_max * d
    data_uniform = np.random.uniform(low, high, amount_of_uniform_numbers)

    # normal distribution parameters
    mean = 0
    sigma = 1
    amount_of_normal_numbers = K_max * (1 + d + d * (d - 1) // 2) + N_max * d
    data_normal = np.random.normal(mean, sigma, amount_of_normal_numbers)

    return generate(data_uniform, data_normal, d, k, n)


def main(d, k, n):
    return generator(d, k, n)
