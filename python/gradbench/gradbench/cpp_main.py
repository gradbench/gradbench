import json
import subprocess
import tempfile
from pathlib import Path


def functions(pathname: str):
    """
    Helper functions for implementing ADBench evals in C++ tools.

    When implementing a tool using C++, for each ADBench eval `$EVAL`,
    you should make a file `tools/$TOOL/$EVAL.py` with these contents:

    ```
    from gradbench.cpp_main import functions

    globals().update(functions(__file__))
    ```
    """

    path = Path(pathname)
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

    def objective(input):
        with tempfile.NamedTemporaryFile("w") as tmp:
            json.dump(input, tmp)
            tmp.flush()
            return subprocess.run(
                [f"tools/{TOOL}/run_{EVAL}", tmp.name, "F"],
                text=True,
                capture_output=True,
            )

    def jacobian(input):
        with tempfile.NamedTemporaryFile("w") as tmp:
            json.dump(input, tmp)
            tmp.flush()
            return subprocess.run(
                [f"tools/{TOOL}/run_{EVAL}", tmp.name, "J"],
                text=True,
                capture_output=True,
            )

    return {
        "compile": compile,
        "objective": objective,
        "jacobian": jacobian,
    }
