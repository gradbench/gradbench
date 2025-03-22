FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# `COPY` the minimal set of files for `uv sync`.
COPY pyproject.toml uv.lock ./
COPY python/gradbench/pyproject.toml python/gradbench/
COPY python/gradbench/gradbench/__init__.py python/gradbench/gradbench/
RUN uv sync --no-default-groups --group gradbench --group numpy --group pydantic

# `COPY` the rest of the files.
COPY python python

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/evals/hello/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
