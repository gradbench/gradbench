FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y autoconf automake build-essential libtool wget

# Install ADOL-C.
RUN wget https://github.com/coin-or/ADOL-C/archive/refs/tags/releases/2.7.2.tar.gz
RUN tar xvf 2.7.2.tar.gz
RUN cd ADOL-C-releases-2.7.2 && autoreconf -fi
RUN cd ADOL-C-releases-2.7.2 && ./configure --enable-sparse --prefix=/usr/local
RUN cd ADOL-C-releases-2.7.2 && make install
ENV LIBRARY_PATH=/usr/local/lib64
ENV LD_LIBRARY_PATH=/usr/local/lib64

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/adol-c tools/adol-c

ENTRYPOINT ["uv", "run", "--no-sync", "python/gradbench/gradbench/tools/adol-c/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
