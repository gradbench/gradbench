#!/usr/bin/env python3

import json
from pathlib import Path


def main():
    tools = sorted(p.name for p in Path("tools").iterdir())
    # GitHub Actions doesn't like whitespace in the JSON
    print(json.dumps({"tool": tools}, separators=(",", ":")))


if __name__ == "__main__":
    main()
