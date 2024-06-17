#!/usr/bin/env python3

import json
from pathlib import Path


def main():
    tools = sorted(p.name for p in Path("tools").iterdir())
    print(json.dumps({"tools": tools}))


if __name__ == "__main__":
    main()
