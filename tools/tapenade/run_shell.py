import json
import sys
import time
import subprocess
from importlib import import_module

# def resolve(name):
#     functions = import_module("functions.c")
#     return getattr(functions, name)


def run(params):
    vals = [1.0*arg["value"] for arg in params["arguments"]]
    
    # Timing only on tapenade not on C compile and execution
    start = time.perf_counter_ns()
    subprocess.run(['tapenade', '-reverse', '-head', 'square(x,y)', '-output', 'double', 'functions.c'],text=True, capture_output=True)
    end = time.perf_counter_ns()
    subprocess.run(['gcc', '-I/usr/tapenade/ADFirstAidKit/', 'run.c', 'functions.c', 'double_b.c', '-o', 'derivative'])
    
    ret = subprocess.run(['./derivative', str(*vals), str(params["name"])], text=True, capture_output=True)
    return {"return": ret.stdout, "nanoseconds": end - start}


def main():
    cfg = json.load(sys.stdin)
    outputs = [run(params) for params in cfg["inputs"]]
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()

