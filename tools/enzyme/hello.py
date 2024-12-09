import json
import subprocess
import tempfile
from os import listdir


def compile():
    c = subprocess.run(
        ["make", "-C", "tools/enzyme", "-B", "-j", "run_hello"],
        stdout=2
    )
    return c.returncode == 0


def square(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/enzyme/run_hello", tmp.name, "F"], text=True, capture_output=True
        )


def double(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/enzyme/run_hello", tmp.name, "J"], text=True, capture_output=True
        )
