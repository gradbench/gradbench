FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential wget

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/manual tools/manual

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/tools/manual/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
