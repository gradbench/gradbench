import json
import subprocess
import tempfile
from os import listdir

TOOL = os.path.split(os.path.dirname(__file__))[-1]
EVAL = __name__


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
