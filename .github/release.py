#!/usr/bin/env python3

import json
import shutil
from pathlib import Path


def ls(path):
    return (p.name for p in Path(path).iterdir())


def summarize(run):
    implemented = run[0]["response"]["success"]
    return {"status": "implemented" if implemented else "unimplemented"}


def main():
    folder = Path("release")
    folder.mkdir()
    table = []
    for e in sorted(ls("evals")):
        row = []
        for t in sorted(ls("tools")):
            source = Path("run") / f"run-{e}-{t}/log.json"
            run = json.loads(source.read_text())
            cell = {"tool": t} | summarize(run)
            row.append(cell)
            shutil.copy(source, folder / f"run-{e}-{t}.json")
        table.append({"eval": e, "tools": row})
    summary = {"table": table}
    (folder / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()