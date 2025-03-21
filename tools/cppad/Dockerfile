FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential cmake pkg-config wget

# Install CppAD.
RUN wget https://github.com/coin-or/CppAD/archive/refs/tags/20250000.2.tar.gz
RUN tar xvf 20250000.2.tar.gz
RUN mkdir CppAD-20250000.2/build
RUN cd CppAD-20250000.2/build && cmake ..
RUN make -C CppAD-20250000.2/build install
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/cppad tools/cppad

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/tools/cppad/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
