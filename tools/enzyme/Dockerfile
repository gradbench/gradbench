FROM python:3.11-slim
ARG ENZYME_VER=0.0.168

WORKDIR /home/gradbench

RUN apt-get update

RUN apt-get install -y \
    build-essential \
    python3 \
    wget

RUN apt-get install -y \
    ninja-build \
    cmake \
    libzstd-dev \
    llvm-19 \
    clang-19 \
    lld-19

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

# Install Enzyme
RUN wget https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v${ENZYME_VER}.tar.gz
RUN tar xvf v${ENZYME_VER}.tar.gz && rm -f v${ENZYME_VER}.tar.gz
RUN mkdir enzyme-build
RUN cd enzyme-build && cmake -G Ninja /home/gradbench/Enzyme-${ENZYME_VER}/enzyme -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm/
RUN ninja -C enzyme-build

COPY cpp /home/gradbench/cpp

RUN make -C cpp

COPY tools/enzyme/ /home/gradbench/tools/enzyme

ENTRYPOINT ["python3", "/home/gradbench/tools/enzyme/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
