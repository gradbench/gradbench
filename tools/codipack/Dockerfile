FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential wget

# Install CoDiPack.
RUN wget https://github.com/SciCompKL/CoDiPack/archive/refs/tags/v2.3.1.tar.gz
RUN tar xvf v2.3.1.tar.gz
ENV CPATH=/gradbench/CoDiPack-2.3.1/include

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
COPY tools/codipack tools/codipack

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "codipack"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
