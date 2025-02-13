import json
import subprocess
import tempfile
from os import listdir
from pathlib import Path

path = Path(__file__)
TOOL = path.parent.name
EVAL = path.stem


def compile():
    try:
        subprocess.check_output(
            ["make", "-C", f"tools/{TOOL}", f"run_{EVAL}", "-B"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    else:
        return (True, None)


def cost(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "cost"],
            text=True,
            capture_output=True,
        )


def dir(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            [f"tools/{TOOL}/run_{EVAL}", tmp.name, "dir"],
            text=True,
            capture_output=True,
        )
