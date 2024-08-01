import json
import sys

import data_gen

i = 0


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

    module = "gmm"
    response = define(module=module, source=source)
    if response.get("success"):
        for n in [1000, 10000]:
            for k in [5, 10, 25, 50, 100, 200]:
                input = data_gen.main(2, k, n)  # d k n
                evaluate(module=module, name="calculate_objectiveGMM", input=input)
                evaluate(module=module, name="calculate_jacobianGMM", input=input)


if __name__ == "__main__":
    main()
