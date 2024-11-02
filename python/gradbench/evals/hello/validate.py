#!/usr/bin/env python3

import sys

from gradbench.validate import assert_equal, validate_fixed


def checker(name, input, a, b):
    assert_equal(a, b)


def main():
    validate_fixed(
        raw=sys.stdin.read(),
        module="hello",
        checker=checker,
    )


if __name__ == "__main__":
    main()
