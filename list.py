#!/usr/bin/env python3

import json
from pathlib import Path


def main():
    print(json.dumps(sorted(p.name for p in Path("tools").iterdir())))


if __name__ == "__main__":
    main()
