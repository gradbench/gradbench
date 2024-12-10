import json
import subprocess
import tempfile
from os import listdir


def compile():
    # Nothing to do here. We assume everything is precompiled.
    return True


def objective(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_lstm", tmp.name, "F"], text=True, capture_output=True
        )


def jacobian(input):
    with tempfile.NamedTemporaryFile("w") as tmp:
        json.dump(input, tmp)
        tmp.flush()
        return subprocess.run(
            ["tools/manual/run_lstm", tmp.name, "J"], text=True, capture_output=True
        )
