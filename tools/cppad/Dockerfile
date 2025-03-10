FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget cmake pkg-config

# Install cppad
RUN wget https://github.com/coin-or/CppAD/archive/refs/tags/20250000.2.tar.gz
RUN tar xvf 20250000.2.tar.gz
RUN mkdir CppAD-20250000.2/build
RUN cd CppAD-20250000.2/build && cmake ..
RUN make -C CppAD-20250000.2/build install
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

COPY cpp /gradbench/cpp
RUN make -C cpp

COPY tools/cppad/ /gradbench/tools/cppad

ENTRYPOINT ["python3", "/gradbench/tools/cppad/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
