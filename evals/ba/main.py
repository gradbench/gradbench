import argparse
import json
import sys
from pathlib import Path
from random import Random

import numpy as np

i = 0


def parse(file):
    lines = iter(Path(file).read_text().splitlines())

    n, m, p = [int(v) for v in next(lines).split()]

    one_cam = [float(x) for x in next(lines).split()]
    cams = np.tile(one_cam, (n, 1)).tolist()

    one_X = [float(x) for x in next(lines).split()]
    X = np.tile(one_X, (m, 1)).tolist()

    one_w = float(next(lines))
    w = np.tile(one_w, p).tolist()

    one_feat = [float(x) for x in next(lines).split()]
    feats = np.tile(one_feat, (p, 1)).tolist()

    camIdx = 0
    ptIdx = 0
    obs = []
    for i in range(p):
        obs.append((camIdx, ptIdx))
        camIdx = (camIdx + 1) % n
        ptIdx = (ptIdx + 1) % m

    return {"cams": cams, "X": X, "w": w, "obs": obs, "feats": feats}


def main():
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

    print("Constructing Define", file=sys.stderr)
    module = "ba"
    response = define(module=module, source=source)
    if response.get("success"):
        for datafile in Path("data").iterdir():
            if datafile.is_file():
                input = parse(datafile)
                print(f"calculating objective for {datafile}", file=sys.stderr)
                evaluate(module=module, name="calculate_objectiveBA", input=input)
                print(f"finished calculating jacobian for {datafile}", file=sys.stderr)
                evaluate(module=module, name="calculate_jacobianBA", input=input)
                print(f"finished calculating jacobian for {datafile}", file=sys.stderr)
                exit()


if __name__ == "__main__":
    main()
