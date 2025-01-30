import json
import subprocess
import tempfile
from os import listdir


def compile():
    return (
        subprocess.run(
            ["make", "-C", f"tools/manual", f"run_hello", "-B"],
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
            ["tools/manual/run_hello", tmp.name, "F"], text=True, capture_output=True
        )


def double(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_hello", tmp.name, "J"], text=True, capture_output=True
        )
