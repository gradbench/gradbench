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

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/modules/TensorflowCommon/utils.py


import tensorflow as tf


def to_tf_tensor(ndarray, dtype=tf.float64):
    """Converts the given multidimensional array to a tensorflow tensor.

    Args:
        ndarray (ndarray-like): parameter for conversion.
        dtype (type, optional): defines a type of tensor elements. Defaults to
            tf.float64.

    Returns:
        tensorflow tensor
    """

    return tf.convert_to_tensor(ndarray, dtype=dtype)


def shape(tf_tensor):
    """Returns shape of a tensorflow tensor like a list if integers."""

    return tf_tensor.get_shape().as_list()


def flatten(tf_tensor, column_major=False):
    """Returns the flaten tensor."""

    if column_major:
        tf_tensor = tf.transpose(tf_tensor)

    return tf.reshape(tf_tensor, [-1])
