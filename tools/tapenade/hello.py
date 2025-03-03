import subprocess


def compile():
    try:
        # generate AD code
        result = subprocess.run(
            [
                "tapenade_3.16/bin/tapenade",
                "-reverse",
                "-head",
                "square(x)\\y",
                "-outputdirectory",
                "tools/tapenade",
                "-output",
                "double",
                "tools/tapenade/functions.c",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return (False, "tapenade failed")

        # compile AD code
        result = subprocess.run(
            [
                "gcc",
                "-Itapenade_3.16/ADFirstAidKit/",
                "tools/tapenade/run_deriv.c",
                "tools/tapenade/functions.c",
                "tools/tapenade/double_b.c",
                "-o",
                "tools/tapenade/derivative",
            ]
        )
        if result.returncode != 0:
            return (False, "gcc failed")

        # compile original code
        result = subprocess.run(
            [
                "gcc",
                "tools/tapenade/run_origin.c",
                "tools/tapenade/functions.c",
                "-o",
                "tools/tapenade/normal",
            ]
        )
        if result.returncode != 0:
            return (False, "gcc failed")

        return (True, None)

    except Exception:
        # print(f"Compilation failed with error {e}", file=sys.stderr)
        return False


def double(vals):
    ret = subprocess.run(
        ["tools/tapenade/derivative", str(vals)], text=True, capture_output=True
    )
    return ret


def square(vals):
    ret = subprocess.run(
        ["tools/tapenade/normal", str(vals)], text=True, capture_output=True
    )
    return ret
