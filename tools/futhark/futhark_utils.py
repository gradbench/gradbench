import re


def get_runtime_ns(ls):
    for l in ls:
        m = re.match("runtime: ([0-9]+)", l)
        if m:
            return int(m.group(1)) * 1000
    raise Exception(
        "Output does not contain runtime information:\n{}".format("\n".join(ls))
    )


def run(server, entry_name, inputs, min_runs, min_seconds):
    timings = []
    elapsed = 0
    while len(timings) < min_runs or elapsed < min_seconds:
        if len(timings) > 0:
            server.cmd_free("out")
        log = server.cmd_call(
            entry_name,
            "out",
            *inputs,
        )
        ns = get_runtime_ns(log)
        timings.append(ns)
        elapsed += ns / 1e9
    vals = server.get_value("out")
    server.cmd_free("out")
    return (vals, timings)
