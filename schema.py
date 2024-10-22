#!/usr/bin/env python3

from pathlib import Path

import yaml


def main():
    print(yaml.safe_load(Path("types.yml").read_text()))
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema",
        "title": "GradBench IR",
    }


if __name__ == "__main__":
    main()
