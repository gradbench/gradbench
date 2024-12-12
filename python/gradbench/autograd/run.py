import json
import sys
from importlib import import_module

from gradbench.wrap import Wrapped


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func: Wrapped = resolve(params["module"], params["function"])
    return func.wrapped(params["input"])


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            try:
                import_module(message["module"])
                response["success"] = True
            except:
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
