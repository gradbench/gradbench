import json
import subprocess
import tempfile
from os import listdir


def compile():
    # Nothing to do here. We assume everything is precompiled.
    return True


def calculate_objectiveBA(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_ba", tmp.name, "F"], text=True, capture_output=True
        )


def calculate_jacobianBA(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_ba", tmp.name, "J"], text=True, capture_output=True
        )
