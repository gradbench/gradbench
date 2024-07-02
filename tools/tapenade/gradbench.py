import subprocess
import time


def double(vals):
    subprocess.run(
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
    subprocess.run(
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
    ret = subprocess.run(["./derivative", str(vals)], text=True, capture_output=True)

    return ret


def square(vals):

    subprocess.run(["gcc", "run_origin.c", "functions.c", "-o", "normal"])
    ret = subprocess.run(["./normal", str(vals)], text=True, capture_output=True)

    return ret
