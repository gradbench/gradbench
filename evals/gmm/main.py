import json
import sys
from pathlib import Path

import data_gen

i = 0


def main():
    source = Path("gmm.adroit").read_text()

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

    module = "gmm"
    response = define(module=module, source=source)
    if response.get("success"):
        for n in [1000, 10000]:
            for k in [5, 10, 25, 50, 100, 200]:
                input = data_gen.main(2, k, n)  # d k n
                workload = f"2_{k}_{n}"
                evaluate(
                    module=module,
                    name="calculate_objectiveGMM",
                    workload=workload,
                    input=input,
                )
                evaluate(
                    module=module,
                    name="calculate_jacobianGMM",
                    workload=workload,
                    input=input,
                )


if __name__ == "__main__":
    main()
