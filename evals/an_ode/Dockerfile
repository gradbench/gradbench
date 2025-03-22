FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential wget

# `COPY` the minimal set of files for `uv sync`.
COPY pyproject.toml uv.lock ./
COPY python/gradbench/pyproject.toml python/gradbench/
COPY python/gradbench/gradbench/__init__.py python/gradbench/gradbench/
RUN uv sync --no-default-groups --group dataclasses-json --group gradbench --group numpy --group pydantic

# Build the `manual` tool.
COPY cpp cpp
RUN make -C cpp
COPY tools/manual tools/manual
RUN make -C tools/manual -Bj

# `COPY` the rest of the files.
COPY python python

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/evals/an_ode/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
