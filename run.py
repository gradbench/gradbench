#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
import time


def run(cmd):
    return subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--tool", required=True)
    args = parser.parse_args()

    server = run(args.tool)
    client = run(args.eval)

    print("[")
    first = True
    for message in client.stdout:
        if not first:
            print(",")
        first = False
        print("  {")
        print(f'    "message": {message.strip()},')
        server.stdin.write(message)
        server.stdin.flush()
        start = time.perf_counter_ns()
        response = server.stdout.readline()
        end = time.perf_counter_ns()
        print(f'    "nanoseconds": {end - start}', end="")
        if server.poll() is not None:
            print()
            print("  }", end="")
            break
        print(",")
        print(f'    "response": {response.strip()}')
        client.stdin.write(response)
        client.stdin.flush()
        print("  }", end="")
    print()
    print("]")
    sys.exit((server.poll() or 0) | (client.poll() or 0))


if __name__ == "__main__":
    main()
