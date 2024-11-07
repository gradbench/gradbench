#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def eprint(s):
    print(s, file=sys.stderr)


def main():
    any_errors = False
    log = json.loads(Path("log.json").read_text())
    for validation in log[-1]["message"]["validations"]:
        if not validation["correct"]:
            any_errors = True
            eprint(f"- id: {validation['id']}")
            eprint("  error: |")
            for line in validation["error"].splitlines():
                eprint(f"    {line}")
    sys.exit(any_errors)


if __name__ == "__main__":
    main()
