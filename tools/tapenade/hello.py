import subprocess
import sys
import time


def compile():
    try:
        # generate AD code
        result = subprocess.run(
            [
                "tapenade",
                "-reverse",
                "-head",
                "square(x)\\y",
                "-output",
                "double",
                "functions.c",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return False

        # compile AD code
        result = subprocess.run(
            [
                "gcc",
                "-I/home/gradbench/tapenade_3.16/ADFirstAidKit/",
                "run_deriv.c",
                "functions.c",
                "double_b.c",
                "-o",
                "derivative",
            ]
        )
        if result.returncode != 0:
            return False

        # compile original code
        result = subprocess.run(["gcc", "run_origin.c", "functions.c", "-o", "normal"])
        if result.returncode != 0:
            return False

        return True

    except Exception as e:
        # print(f"Compilation failed with error {e}", file=sys.stderr)
        return False


def double(vals):
    ret = subprocess.run(["./derivative", str(vals)], text=True, capture_output=True)
    return ret


def square(vals):
    ret = subprocess.run(["./normal", str(vals)], text=True, capture_output=True)
    return ret
