FROM debian:trixie
ARG ENZYME_VER=0.0.172
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget \
    git
RUN apt-get update && apt-get install -y \
    ninja-build \
    cmake \
    libzstd-dev \
    llvm-19 \
    clang-19 \
    libclang-19-dev \
    lld-19 \
    libomp-dev

# Set clang.
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100
RUN update-alternatives --install /usr/bin/lld lld /usr/bin/lld-19 100

# Install Enzyme.
RUN wget https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v${ENZYME_VER}.tar.gz
RUN tar xvf v${ENZYME_VER}.tar.gz && rm -f v${ENZYME_VER}.tar.gz
RUN mkdir enzyme-build
RUN cd enzyme-build && cmake -G Ninja /gradbench/Enzyme-${ENZYME_VER}/enzyme -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm/ -DClang_DIR=/usr/lib/llvm-19/lib/cmake/clang/
RUN ninja -C enzyme-build

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/enzyme tools/enzyme

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "enzyme"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
