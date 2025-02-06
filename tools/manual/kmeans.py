import json
import subprocess
import tempfile
from os import listdir
from pathlib import Path

path = Path(__file__)
TOOL = path.parent.name
EVAL = path.stem


def compile():
    return (
        subprocess.run(
            ["make", "-C", f"tools/{TOOL}", f"run_{EVAL}", "-B"],
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )


def cost(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_kmeans", tmp.name, "cost"],
            text=True,
            capture_output=True,
        )


def dir(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_kmeans", tmp.name, "dir"], text=True, capture_output=True
        )
