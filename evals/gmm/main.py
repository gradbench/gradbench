import argparse
import json
import sys
from pathlib import Path
from random import Random


def parse(datafile):
    lines = iter(Path(datafile).read_text().splitlines())
    d, k, n = [int(v) for v in next(lines).split()]
    alpha = []
    for _ in range(k):
        alpha.append(float(next(lines)))
    means = []
    for _ in range(k):
        means.append([float(v) for v in next(lines).split()])
    icf = []
    for _ in range(k):
        icf.append([float(v) for v in next(lines).split()])
    x = []
    for _ in range(n):
        x.append([float(v) for v in next(lines).split()])
    last = next(lines).split()
    gamma = float(last[0])
    m = int(last[1])
    return {
        "d": d,
        "k": k,
        "n": n,
        "alpha": alpha,
        "means": means,
        "icf": icf,
        "x": x,
        "gamma": gamma,
        "m": m,
    }


i = 0


def main():
    # this is where I would link adroit file for gmm functions
    # source = Path("gradbench.adroit").read_text()
    source = "PLACE HOLDER"

    def send(message):
        global i
        json.dump({"id": i} | message, sys.stdout)
        print(flush=True)
        response = json.loads(sys.stdin.readline())
        if response["id"] != i:
            raise ValueError(f"expected message ID {i}, got {response['id']}")
        i += 1
        return response

    def define(*, module, source):
        return send({"kind": "define", "module": module, "source": source})

    def evaluate(*, module, name, input):
        return send(
            {"kind": "evaluate", "module": module, "name": name, "input": input}
        )

    module = "gmm"
    response = define(module=module, source=source)
    if response.get("success"):
        files = ["d2_k5.txt"]
        for file in files:
            x = parse(file)
            evaluate(module=module, name="calculate_objectiveGMM", input=x)["output"]
            evaluate(module=module, name="calculate_jacobianGMM", input=x)["output"]


if __name__ == "__main__":
    main()
