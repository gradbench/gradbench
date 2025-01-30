# Generic module for suitable for use by all tools that make use
# of the command line API provided by cpp/adbench/main.h.
#
# For each eval 'foo', you should make a relative symlink 'foo.py' to
# this file.

import json
import os
import subprocess
import tempfile

TOOL = os.path.split(os.path.dirname(__file__))[-1]
EVAL = __name__


def compile():
    return (
        subprocess.run(
            ["make", "-C", f"tools/{TOOL}", f"run_{EVAL}", "-B"],
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )


def objective(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "F"], text=True, capture_output=True
        )


def jacobian(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "J"], text=True, capture_output=True
        )
