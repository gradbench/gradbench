FROM python:3.11-slim

WORKDIR /home/gradbench

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

copy python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python/gradbench

COPY cpp /home/gradbench/cpp

RUN make -C cpp

COPY tools/cppad/ /home/gradbench/tools/cppad

RUN make -j -B -C tools/cppad

ENTRYPOINT ["python3", "/home/gradbench/tools/cppad/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
