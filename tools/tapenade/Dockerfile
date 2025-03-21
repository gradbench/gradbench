FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y bash build-essential openjdk-17-jre-headless wget

# Install Tapenade.
RUN wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
RUN tar -xvzf tapenade_3.16.tar

# Ensure Tapenade is accessible.
RUN ln -s tapenade_3.16/bin/tapenade /usr/local/bin/tapenade

# `COPY` the minimal set of files for `uv sync`.
COPY pyproject.toml uv.lock ./
COPY python/gradbench/pyproject.toml python/gradbench/
COPY python/gradbench/gradbench/__init__.py python/gradbench/gradbench/
RUN uv sync --no-default-groups --group gradbench

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/tapenade tools/tapenade

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/tools/tapenade/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
