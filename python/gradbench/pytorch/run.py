import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path


def resolve(module, name):
    functions = import_module(module)
    return getattr(functions, name)


def run(params):
    func = resolve(params["module"], params["name"])
    input = func.prepare(params["input"])
    start = time.perf_counter_ns()
    ret = func(input)
    end = time.perf_counter_ns()
    return {"output": func.unwrap(ret), "nanoseconds": {"evaluate": end - start}}


def adroit_path(module: str) -> Path:
    folder = Path(__file__).parent / "tmp/adroit"
    path = folder / f"{module}.adroit"
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    if path.is_relative_to(folder):
        return path
    else:
        raise ValueError(f"resolved path {path} is not relative to {folder}")


def define(module, source):
    try:  # File already exists
        import_module(module)
    except:  # Try to translate to create file
        src = adroit_path(module)
        src.write_text(source)
        adroit = "usr/local/bin/adroit"

        subprocess.run(
            f"{adroit} json {src} > hello.json",
            shell=True,
        )

        # generates module to import
        subprocess.run(
            [
                "python3",
                Path(__file__).parent / "translator.py",
                module,
            ]
        )
        import_module(module)


def main():
    for line in sys.stdin:
        message = json.loads(line)
        response = {}
        if message["kind"] == "evaluate":
            response = run(message)
        elif message["kind"] == "define":
            try:
                define(message["module"], message["source"])
                response["success"] = True
            except:
                response["success"] = False
        print(json.dumps({"id": message["id"]} | response), flush=True)


if __name__ == "__main__":
    main()
