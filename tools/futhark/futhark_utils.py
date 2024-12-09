import re


def get_runtime_ns(ls):
    for l in ls:
        m = re.match("runtime: ([0-9]+)", l)
        if m:
            return int(m.group(1)) * 1000


def run(server, entry_name, outputs, inputs, runs):
    times = []
    for i in range(runs):
        if i > 0:
            for o in outputs:
                server.cmd_free(o)
        out = server.cmd_call(
            entry_name,
            *outputs,
            *inputs,
        )
        times += [get_runtime_ns(out)]
    return (
        tuple([server.get_value(o) for o in outputs]),
        times,
    )
