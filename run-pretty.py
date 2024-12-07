#!/usr/bin/env python3
#
# An intermediary like run.py, except where run.py prints the raw JSON
# messages, run-pretty.py produces concise human-readable output, with
# one line per definition/evaluation, and checkmarks to indicate
# successful validation.
#
# A machine-readable log file containing all messages can also be
# produced with the --json option.

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from termcolor import colored


def run(cmd):
    return subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def tag(id):
    return colored(f"[{id:4d}]", "cyan", attrs=["bold"])


def message_define(message):
    module = message["module"]
    id = message["id"]
    print(f"{tag(id)} Defining module {module}... ", end="")
    sys.stdout.flush()

    def f(response):
        if response["success"]:
            print("Victory!")
        else:
            sys.exit(1)

    return f


def message_evaluate(log, client, message):
    module = message["module"]
    name = message["name"]
    id = message["id"]
    print(f"{tag(id)} Eval {name:25} {message['workload']:15} ", end="")
    sys.stdout.flush()

    def f(response):
        ns = response["nanoseconds"]["evaluate"]
        print(f"{ns:10} ns ", end="")
        sys.stdout.flush()
        analysis = json.loads(client.stdout.readline())
        log.write(",\n")
        log.write(f'    "analysis": {json.dumps(analysis)}')
        valid = analysis["correct"]
        print(colored("✓", "green") if valid else colored("⚠", "red"))
        if not valid:
            print(colored(analysis["error"], "red"))

    return f


def message_end(message):
    sys.exit(0)


def on_message(log, client, message):
    if message["kind"] == "define":
        return message_define(message)
    elif message["kind"] == "end":
        return message_end(message)
    elif message["kind"] == "evaluate":
        return message_evaluate(log, client, message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--json", required=False)
    args = parser.parse_args()

    if args.json:
        print(f"Writing log to {args.json}.")
        log = open(args.json, "w")
    else:
        log = open(os.devnull, "w")

    log.write("[\n")

    server = run(args.tool)
    client = run(args.eval)

    first = True
    for message in client.stdout:
        if not first:
            log.write(",\n")
        first = False

        log.write("  {\n")
        log.write(f'    "message": {message.strip()}')

        message_json = json.loads(message)

        if json.loads(message)["kind"] == "end":
            log.write("\n  }")
            break
        log.write(",\n")

        on_response = on_message(log, client, message_json)
        server.stdin.write(message)
        server.stdin.flush()
        response = server.stdout.readline()

        log.write(f'    "response": {response.strip()}')

        client.stdin.write(response)
        client.stdin.flush()

        if on_response != None:
            response_json = json.loads(response)
            on_response(response_json)

        log.write("  }")
        sys.stdout.flush()
    log.write("\n]")
    sys.exit((server.poll() or 0) | (client.poll() or 0))


if __name__ == "__main__":
    main()
