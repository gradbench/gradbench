import json
import sys


def main():
    with open("hello.json", "r") as hello:
        data = json.load(hello)
        return parse(data)


# TODO: Actually have this parse the json
def parse(data):
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


def generate_file(code):
    with open("/home/gradbench/python/gradbench/pytorch/hello_t.py", "w") as file:
        file.write(code)


if __name__ == "__main__":
    code = main()
    generate_file(code)
