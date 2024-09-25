import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from gradbench.adroit import Modules
from gradbench.pytorch import gmm, hello
from gradbench.wrap_module import Functions


def hardcoded() -> dict[str, Any]:
    return {"hello": hello, "gmm": gmm}


def adroit_path(module: str) -> Path:
    folder = Path(__file__).parent / "tmp/adroit"
    path = folder / f"{module}.adroit"
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    if path.is_relative_to(folder):
        return path
    else:
        raise ValueError(f"resolved path {path} is not relative to {folder}")


def define(module: str, cache: dict[str, Any], source: str) -> Any:
    src = adroit_path(module)
    src.write_text(source)
    mod = cache.get(module)
    if mod is not None:
        return mod
    ir = Modules.model_validate_json(subprocess.check_output(["adroit", "json", src]))


def main() -> None:
    cache = hardcoded()
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        match message["kind"]:
            case "define":
                module = message["module"]
                mod = define(module, cache, message["source"])
                if mod is None:
                    response = {"success": False}
                else:
                    cache[module] = mod
                    response = {"success": True}
            case "evaluate":
                func: Functions = getattr(cache[message["module"]], message["name"])
                input = func.prepare(message["input"])
                start = time.perf_counter_ns()
                ret = func(input)
                end = time.perf_counter_ns()
                nanos = {"evaluate": end - start}
                response = {"output": func.unwrap(ret), "nanoseconds": nanos}
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
