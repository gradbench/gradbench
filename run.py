#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
import time
import os
import json


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
    parser.add_argument("--results", type=str, default="results", metavar="DIR")
    args = parser.parse_args()

    server = run(args.tool)
    client = run(args.eval)

    os.makedirs(args.results, exist_ok=True)

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

        message_json = json.loads(message)
        response_json = json.loads(response)

        if message_json.get("kind") == "evaluate":
            results_dir = os.path.join(
                args.results,
                message_json["module"],
                response_json["tool"],
                message_json["name"],
                message_json["workload"],
            )
            os.makedirs(results_dir, exist_ok=True)
            input_fname = os.path.join(results_dir, "input")
            output_fname = os.path.join(results_dir, "output")
            nanoseconds_fname = os.path.join(results_dir, "nanoseconds.json")
            json.dump(message_json["input"], open(input_fname, "w"))
            json.dump(response_json["output"], open(output_fname, "w"))
            json.dump(response_json["nanoseconds"], open(nanoseconds_fname, "w"))

    print()
    print("]")
    sys.exit((server.poll() or 0) | (client.poll() or 0))


if __name__ == "__main__":
    main()
