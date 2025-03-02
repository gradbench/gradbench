# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/input_utils.py

"""
Changes made:
- Removed everything not related to the LSTM benchmark.
"""

import numpy as np

from gradbench.adbench.lstm_data import LSTMInput


def parse_floats(arr):
    """Parses enumerable as float.

    Args:
        arr (enumerable): input data that can be parsed to floats.

    Returns:
        (List[float]): parsed data.
    """

    return [float(x) for x in arr]


def read_lstm_instance(fn):
    """Reads input data for LSTM objective from the given file.

    Args:
        fn (str): input file name.

    Returns:
        (LSTMInput): input data for LSTM objective test class.
    """

    fid = open(fn)

    line = fid.readline().split()
    layer_count = int(line[0])
    char_count = int(line[1])

    fid.readline()
    main_params = np.array(
        [parse_floats(fid.readline().split()) for _ in range(2 * layer_count)]
    )

    fid.readline()
    extra_params = np.array([parse_floats(fid.readline().split()) for _ in range(3)])

    fid.readline()
    state = np.array(
        [parse_floats(fid.readline().split()) for _ in range(2 * layer_count)]
    )

    fid.readline()
    text_mat = np.array(
        [parse_floats(fid.readline().split()) for _ in range(char_count)]
    )

    fid.close()

    return LSTMInput(main_params, extra_params, state, text_mat)
