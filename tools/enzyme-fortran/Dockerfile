FROM debian:trixie
ARG ENZYME_VER=0.0.311
ARG LLVM_VER=22
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
    llvm-${LLVM_VER}-dev \
    flang-${LLVM_VER} \
    libflang-${LLVM_VER}-dev

# Install Enzyme with the flang pass plugin and the Fortran bindings
# (`use enzyme`). The bindings are compiled with flang, so that the installed
# .mod files are readable by the flang that compiles the tool.
RUN wget https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v${ENZYME_VER}.tar.gz
RUN tar xf v${ENZYME_VER}.tar.gz && rm -f v${ENZYME_VER}.tar.gz
RUN cmake -G Ninja -S /gradbench/Enzyme-${ENZYME_VER}/enzyme -B enzyme-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/enzyme \
    -DLLVM_DIR=/usr/lib/llvm-${LLVM_VER}/lib/cmake/llvm \
    -DCMAKE_Fortran_COMPILER=/usr/lib/llvm-${LLVM_VER}/bin/flang \
    -DENZYME_CLANG=OFF \
    -DENZYME_FLANG=ON \
    -DENZYME_FORTRAN=ON
RUN ninja -C enzyme-build install && rm -rf enzyme-build Enzyme-${ENZYME_VER}

ENV FLANG_PREFIX=/usr/lib/llvm-${LLVM_VER}
ENV ENZYME_PREFIX=/opt/enzyme
ENV LLVM_VERSION=${LLVM_VER}
ENV LD_LIBRARY_PATH=/opt/enzyme/lib

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/enzyme-fortran tools/enzyme-fortran

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "enzyme-fortran"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
