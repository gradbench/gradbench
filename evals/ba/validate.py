#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from gradbench.validate import validate_fixed


def checker(name, input, a, b):
    match name:
        case "calculate_objectiveBA":
            assert np.all(np.isclose(a, b))
        case "calculate_jacobianBA":
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    validate_fixed(
        module="ba",
        golden=Path(args.golden).read_text(),
        log=Path(args.log).read_text(),
        checker=checker,
    )


if __name__ == "__main__":
    main()
