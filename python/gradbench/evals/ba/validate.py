#!/usr/bin/env python3

import sys

import numpy as np

from gradbench.validate import validate_fixed


def checker(name, input, a, b):
    match name:
        case "calculate_objectiveBA":
            assert (
                np.all(
                    np.isclose(
                        a["reproj_error"]["elements"], b["reproj_error"]["elements"]
                    )
                )
                and a["reproj_error"]["repeated"] == b["reproj_error"]["repeated"]
                and np.all(np.isclose(a["w_err"]["element"], b["w_err"]["element"]))
                and a["w_err"]["repeated"] == b["w_err"]["repeated"]
            )
        case "calculate_jacobianBA":
            assert a == b


def main():
    validate_fixed(
        raw=sys.stdin.read(),
        module="ba",
        checker=checker,
    )


if __name__ == "__main__":
    main()
