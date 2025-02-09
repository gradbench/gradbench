#!/usr/bin/env python3

import sys

targets = {
    "amd64": "x86_64-unknown-linux-musl",
    "arm64": "aarch64-unknown-linux-musl",
}
print(targets[sys.argv[1]])
