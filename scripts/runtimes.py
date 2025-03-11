#!/usr/bin/env python3
#
# Print information about the runtime measurements in the given log file.

import sys
import json
import numpy as np

fname = sys.argv[1]

module = None
function = None
description = None


def on_json(j):
    global description, function, module
    if "message" in j and j["message"]["kind"] == "evaluate":
        module = j["message"]["module"]
        function = j["message"]["function"]
        description = j["message"]["description"]
    if "response" in j and "timings" in j["response"]:
        print(f"\n{module}::{function}({description})")
        timings = {}
        for x in j["response"]["timings"]:
            if x["name"] not in timings:
                timings[x["name"]] = []
            timings[x["name"]].append(x["nanoseconds"]/1000)
        for t in timings:
            mean = np.mean(timings[t])
            stddev = np.std(timings[t])
            print(f"  {t}")
            print(f"    avg: {mean}")
            print(f"    max: {np.max(timings[t])}")
            print(f"    min: {np.min(timings[t])}")
            print(f"    dev: {stddev / mean}")


with open(fname, "r") as f:
    for line in f:
        on_json(json.loads(line))
