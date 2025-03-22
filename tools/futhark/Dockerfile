FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
ARG FUTHARK_VER=0.25.29
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential git xz-utils

# `COPY` the minimal set of files for `uv sync`.
COPY pyproject.toml uv.lock ./
COPY python/gradbench/pyproject.toml python/gradbench/
COPY python/gradbench/gradbench/__init__.py python/gradbench/gradbench/
RUN uv sync --no-default-groups --group dataclasses-json --group futhark-server --group gradbench

# Install Futhark.
ADD https://futhark-lang.org/releases/futhark-${FUTHARK_VER}-linux-x86_64.tar.xz futhark-${FUTHARK_VER}-linux-x86_64.tar.xz
RUN tar xvf futhark-${FUTHARK_VER}.tar.xz
RUN make -C futhark-${FUTHARK_VER}-linux-x86_64 install

# Install Futhark packages.
COPY tools/futhark tools/futhark
RUN cd tools/futhark && futhark pkg sync

# `COPY` the rest of the files.
COPY python python

ENTRYPOINT ["uv", "run", "--no-sync", "tools/futhark/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
