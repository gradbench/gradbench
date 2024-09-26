import json
import subprocess
import sys
import time
from importlib import import_module


def resolve(module, name):
    functions = import_module("{}_t".format(module))
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["name"])
    input = func.prepare(params["input"])
    start = time.perf_counter_ns()
    ret = func(input)
    end = time.perf_counter_ns()
    return {"output": func.unwrap(ret), "nanoseconds": {"evaluate": end - start}}


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            # try:
            # import_module(message["module"])

            # generates module to import
            subprocess.run(
                [
                    "python3",
                    "/home/gradbench/python/gradbench/pytorch/translator.py",
                ]
            )
            mod = "{}_t".format(message["module"])
            import_module(mod)
            response["success"] = True
        # except:
        # response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
