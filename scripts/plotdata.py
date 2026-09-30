#!/usr/bin/env python3
#
# Based on the log files produces by 'gradbench', produce data files
# that can be used as input to plot.gnuplot. Three data files
# produced, each with a column per provided JSON file:
#
# * Absolute runtime of primal function.
#
# * Absolute runtime of differentiated function (specifics depend on
#   the eval; typically computing the full Jacobian).
#
# * Relative runtime of the differentiated function compared to the
#   primal, i.e., the overhead of AD.
#
# Used like:
#
#   $ scripts/plotdata.py futhark.jsonl pytorch.jsonl finite.jsonl manual.jsonl tapenade.jsonl
#
# All of the log files must be from evaluations of the same eval.

import argparse
import json
import sys


def mean(xs):
    if len(xs) == 0:
        return 0
    else:
        return sum(xs) / len(xs)


def read_msgs(fname):
    try:
        msgs = []
        with open(fname, "r") as f:
            for line in f:
                msgs.append(json.loads(line))
        if len(msgs) % 2 != 0:
            print(f"{fname} contains an odd number of messages.", file=sys.stderr)
        return list(zip(msgs[0::2], msgs[1::2]))
    except Exception as e:
        print(f"Failed to read {fname}:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("jsons", nargs="+")
args = parser.parse_args()

eval = None
tools = {}
for fname in args.jsons:
    print(f"Reading {fname}... ", end="", file=sys.stderr)
    msgs = read_msgs(fname)
    this_eval = msgs[0][0]["message"]["eval"]
    if "tool" not in msgs[0][1]["response"]:
        print("invalid, skipping.", file=sys.stderr)
        continue
    this_tool = msgs[0][1]["response"]["tool"]
    print(f"tool={this_tool} eval={this_eval}.", file=sys.stderr)
    if eval is None:
        eval = this_eval
    elif this_eval != eval:
        print(
            f'{fname}: is a log for eval "{this_eval}", which differs from "{eval}".',
            file=sys.stderr,
        )
        sys.exit(1)
    tools[this_tool] = msgs

workloads = {}
primal_runtimes = {}
diff_runtimes = {}

KNOWN_EVALS = {
    "gmm": {"primal": "objective", "diff": "jacobian"},
    "lstm": {"primal": "objective", "diff": "jacobian"},
    "ht": {"primal": "objective", "diff": "jacobian"},
    "ba": {"primal": "objective", "diff": "jacobian"},
    "kmeans": {"primal": "cost", "diff": "dir"},
}

primal = KNOWN_EVALS[eval]["primal"]
diff = KNOWN_EVALS[eval]["diff"]

for tool in tools:
    print(f"Extrating measurements for {tool}...", file=sys.stderr)
    primal_runtimes[tool] = {}
    diff_runtimes[tool] = {}
    for e in tools[tool]:
        message = e[0]["message"]
        response = e[1]["response"]
        if message["kind"] == "evaluate":
            workload = message["description"]
            workloads[workload] = None
            if not response["success"]:
                continue
            if message["function"] == primal:
                primal_runtimes[tool][workload] = (
                    mean(
                        [
                            x["nanoseconds"]
                            for x in response["timings"]
                            if x["name"] == "evaluate"
                        ]
                    )
                    / 1e9
                )
            if message["function"] == diff:
                diff_runtimes[tool][workload] = (
                    mean(
                        [
                            x["nanoseconds"]
                            for x in response["timings"]
                            if x["name"] == "evaluate"
                        ]
                    )
                    / 1e9
                )


def gendata(variant, point):
    with open(f"{eval}-{variant}.data", "w") as f:
        f.write("Workload")
        for tool in tools:
            f.write(" ")
            f.write(tool.replace("_", "\\\\_"))
        f.write("\n")

        for w in workloads:
            f.write(w.replace("_", "\\\\_"))
            for tool in tools:
                f.write(" ")
                f.write(point(tool, w))
            f.write("\n")


gendata(
    "primal",
    lambda tool, w: str(primal_runtimes[tool][w])
    if w in primal_runtimes[tool]
    else "?",
)
gendata(
    "diff",
    lambda tool, w: str(diff_runtimes[tool][w]) if w in diff_runtimes[tool] else "?",
)


def ratio_point(tool, w):
    if w in diff_runtimes[tool] and primal_runtimes[tool][w]:
        return str(diff_runtimes[tool][w] / primal_runtimes[tool][w])
    else:
        return "?"


gendata("ratio", ratio_point)

print(eval)
