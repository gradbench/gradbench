import json
import sys
from pathlib import Path


# TODO: Actually have this parse the json
def parse(data, module):
    code = """import torch
from gradbench.wrap_module import wrap

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)
    """
    modules = data["modules"]
    for key, content in modules.items():
        for definition in content["tree"]["defs"]:
            if definition["name"] == 7:
                code += """
@wrap(to_tensor, lambda x: x.item())
def square(x):
    return x * x
                """
            elif definition["name"] == 22:
                code += """
@wrap(to_tensor, lambda x: x.item())
def double(x):
    y = square(x)
    y.backward()
    return x.grad
                """
    return code


def generate_file(code, module):
    with open(f"{Path(__file__).parent}/{module}.py", "w") as file:
        file.write(code)


def main(module):
    with open(f"{module}.json", "r") as mod:
        data = json.load(mod)
        return parse(data, module)


if __name__ == "__main__":
    module = sys.argv[1]
    code = main(module)
    generate_file(code, module)
