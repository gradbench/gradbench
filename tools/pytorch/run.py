import json
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch


@dataclass
class FuncWithConverters:
    func: callable
    wrap_in: callable
    wrap_out: callable


def resolve_ba(name):
    import ba

    return {
        "calculate_objectiveBA": FuncWithConverters(
            func=ba.calculate_objectiveBA,
            wrap_in=ba.parse_input,
            wrap_out=ba.objective_output,
        ),
        "calculate_jacobianBA": FuncWithConverters(
            func=ba.calculate_jacobianBA,
            wrap_in=ba.parse_input,
            wrap_out=ba.jacobian_output,
        ),
    }[name]


def resolve_gmm(name):
    import gmm

    return {
        "calculate_objectiveGMM": FuncWithConverters(
            func=gmm.calculate_objectiveGMM,
            wrap_in=lambda x: x,
            wrap_out=lambda x: x.tolist(),
        ),
        "calculate_jacobianGMM": FuncWithConverters(
            func=gmm.calculate_jacobianGMM,
            wrap_in=lambda x: x,
            wrap_out=lambda x: x.tolist(),
        ),
    }[name]


def resolve_gradbench(name):
    import gradbench

    return {
        "square": FuncWithConverters(
            func=gradbench.square,
            wrap_in=lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True),
            wrap_out=lambda x: x.item(),
        ),
        "double": FuncWithConverters(
            func=gradbench.double,
            wrap_in=lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True),
            wrap_out=lambda x: x.item(),
        ),
    }[name]


def resolve(module, name):
    return {"ba": resolve_ba, "gmm": resolve_gmm, "gradbench": resolve_gradbench}[
        module
    ](name)


def run(params):
    functions = resolve(params["module"], params["name"])
    arg = functions.wrap_in(params["input"])
    start = time.perf_counter_ns()
    ret = functions.func(arg)
    end = time.perf_counter_ns()
    return {"output": functions.wrap_out(ret), "nanoseconds": {"evaluate": end - start}}


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
