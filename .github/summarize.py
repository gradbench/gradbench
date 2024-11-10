#!/usr/bin/env python3

import json
from pathlib import Path


def ls(path):
    return (p.name for p in Path(path).iterdir())


def summarize(run):
    implemented = run[0]["response"]["success"]
    if not implemented:
        return {"status": "unimplemented"}
    validations = run[-1]["message"]["validations"]
    correct = all(validation["correct"] for validation in validations)
    return {"status": "correct" if correct else "incorrect"}


def main():
    folder = Path("run")
    table = []
    for e in sorted(ls("evals")):
        row = []
        for t in sorted(ls("tools")):
            source = folder / f"run-{e}-{t}/log.json"
            run = json.loads(source.read_text())
            cell = {"tool": t} | summarize(run)
            row.append(cell)
        table.append({"eval": e, "tools": row})
    summary = {"table": table}
    (folder / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
