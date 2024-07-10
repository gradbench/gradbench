import json
import sys
from pathlib import Path
from random import Random


class IdGenerator:
    rand: Random
    used: set[int]

    def __init__(self, *, seed):
        self.rand = Random(seed)
        self.used = set()

    def attempt(self):
        return self.rand.randrange(0, 2**31)

    def __call__(self):
        i = self.attempt()
        while i in self.used:
            i = self.attempt()
        self.used.add(i)
        return i


def main():
    source = Path("gradbench.adroit").read_text()

    id_gen = IdGenerator(seed=0)

    def send(message):
        i = id_gen()
        json.dump({"id": i} | message, sys.stdout)
        print(flush=True)
        response = json.loads(sys.stdin.readline())
        if response["id"] != i:
            raise ValueError(f"expected message ID {i}, got {response['id']}")
        return response

    def define(*, module, source):
        return send({"kind": "define", "module": module, "source": source})

    def evaluate(*, module, name, input):
        return send(
            {"kind": "evaluate", "module": module, "name": name, "input": input}
        )

    module = "gradbench"
    response = define(module=module, source=source)
    if response.get("success"):
        x = 1.0
        for _ in range(4):
            y = evaluate(module=module, name="square", input=x)["output"]
            x = evaluate(module=module, name="double", input=y)["output"]


if __name__ == "__main__":
    main()
