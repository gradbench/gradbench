#!/usr/bin/env python3

import json
from datetime import datetime, timezone
from pathlib import Path


def ls(path):
    return sorted(p.name for p in Path(path).iterdir())


def output(name, value):
    # GitHub Actions doesn't like whitespace in the JSON
    print(f"{name}={json.dumps(value, separators=(',', ':'))}")


def main():
    output("date", str(datetime.now(timezone.utc).date()))
    output("eval", ls("evals"))
    output("tool", ls("tools"))


if __name__ == "__main__":
    main()
