#!/usr/bin/env python3

import sys

import numpy as np

from gradbench.validate import validate_fixed


def checker(name, input, a, b):
    assert np.all(np.isclose(a, b))


def main():
    validate_fixed(
        raw=sys.stdin.read(),
        module="gmm",
        checker=checker,
    )


if __name__ == "__main__":
    main()
