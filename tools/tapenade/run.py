import json
import sys
import time
from importlib import import_module

# NOTE: The current implementation for Tapenade only supports passing in individual scalar inputs.


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["name"])
    vals = 1.0 * params["input"]

    # start = time.perf_counter_ns()
    ret, time = func(vals).stdout.split(" ")
    # end = time.perf_counter_ns()

    return {"output": float(ret), "nanoseconds": {"evaluate": int(time)}}


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
