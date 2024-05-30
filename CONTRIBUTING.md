# Contributing to GradBench

## Prerequisites

Make sure to have these tools installed:

- [Git](https://git-scm.com/downloads)
- [Rust](https://www.rust-lang.org/tools/install)
- [Python](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)

## Setup

Once you've installed all prerequisites, clone this repo.

```sh
git clone https://github.com/gradbench/gradbench
```

Then open a terminal in your clone of it; for instance, if you cloned it via the
terminal, run this command:

```sh
cd gradbench
```

Then you can use the `gradbench.sh` script to build GradBench from source and
run it:

```sh
./gradbench.sh help
```

## Python

The Python tools you to set up the Poetry project first:

```sh
poetry install
```

This doesn't put anything on the `PATH` by default. When running a command, you
can put the Python scripts on the `PATH` via `poetry run`; for instance:

```sh
poetry run ./gradbench.sh pytorch tools.pytorch.constant big
```
