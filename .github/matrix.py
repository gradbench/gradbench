#!/usr/bin/env python3

import json
from pathlib import Path


def ls(path):
    return sorted(p.name for p in Path(path).iterdir())


def output(name, value):
    # GitHub Actions doesn't like whitespace in the JSON
    print(f"{name}={json.dumps(value, separators=(',', ':'))}")


def main():
    output("eval", ls("evals"))
    omit = {"diffsharp", "scilean", "tensorflow", "zygote"}
    output("tool", [tool for tool in ls("tools") if tool not in omit])


if __name__ == "__main__":
    main()
