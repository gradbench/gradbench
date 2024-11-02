#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from gradbench.validate import validate_fixed


def checker(name, input, a, b):
    assert np.all(np.isclose(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    validate_fixed(
        module="gmm",
        golden=Path(args.golden).read_text(),
        log=Path(args.log).read_text(),
        checker=checker,
    )


if __name__ == "__main__":
    main()
