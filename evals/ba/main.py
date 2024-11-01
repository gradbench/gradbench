import json
import sys
from pathlib import Path

i = 0


def parse(file):
    lines = iter(Path(file).read_text().splitlines())

    n, m, p = [int(v) for v in next(lines).split()]

    one_cam = [float(x) for x in next(lines).split()]

    one_X = [float(x) for x in next(lines).split()]

    one_w = float(next(lines))

    one_feat = [float(x) for x in next(lines).split()]

    return {
        "n": n,
        "m": m,
        "p": p,
        "cam": one_cam,
        "x": one_X,
        "w": one_w,
        "feat": one_feat,
    }


def main():
    # source = Path("hello.adroit").read_text()
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

    def evaluate(*, module, name, workload, input):
        return send(
            {
                "kind": "evaluate",
                "module": module,
                "name": name,
                "workload": workload,
                "input": input,
            }
        )

    module = "ba"
    response = define(module=module, source=source)
    if response.get("success"):
        # NOTE: data files are taken directly from ADBench. See README for more information.
        # Currently set to run on the smallest two data files. To run on all 20 set loop range to be: range(1,21)
        for i in range(1, 3):
            datafile = next(Path("data").glob(f"ba{i}_*.txt"), None)
            if datafile:
                input = parse(datafile)
                workload = str(datafile.stem)
                evaluate(
                    module=module,
                    name="calculate_objectiveBA",
                    workload=workload,
                    input=input,
                )
                evaluate(
                    module=module,
                    name="calculate_jacobianBA",
                    workload=workload,
                    input=input,
                )


if __name__ == "__main__":
    main()
