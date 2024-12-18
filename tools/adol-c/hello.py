import json
import subprocess
import tempfile
from os import listdir

TOOL = "adol-c"
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


def square(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "F"], text=True, capture_output=True
        )


def double(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "J"], text=True, capture_output=True
        )
