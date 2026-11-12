FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y autoconf automake build-essential libtool wget libeigen3-dev

# Install autodiff.
RUN wget https://github.com/autodiff/autodiff/archive/refs/tags/v1.1.2.tar.gz
RUN tar xvf v1.1.2.tar.gz
ENV CPATH=/gradbench/autodiff-1.1.2

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/autodiff tools/autodiff

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "autodiff"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
