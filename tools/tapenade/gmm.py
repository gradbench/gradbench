import json
import subprocess
import tempfile
from os import listdir


def compile():
    # Nothing to do here. We assume everything is precompiled.
    return True


def calculate_objectiveGMM(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/tapenade/run_gmm", tmp.name, "F"], text=True, capture_output=True
        )


def calculate_jacobianGMM(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/tapenade/run_gmm", tmp.name, "J"], text=True, capture_output=True
        )
