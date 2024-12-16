import json
import subprocess
import tempfile
from os import listdir


def compile():
    # Nothing to do here. We assume everything is precompiled.
    return True


def square(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/adol-c/run_hello", tmp.name, "F"], text=True, capture_output=True
        )


def double(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/adol-c/run_hello", tmp.name, "J"], text=True, capture_output=True
        )
