FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential cmake wget

# Install XAD v2.2.0: Release build with strong inlining, no thread-local tape, huge pages,
# copy elision, skipped zero partials and reduced tape memory; static, without PIC, since only the
# tool links it. XAD's default SIMD option is NATIVE on arm64, so NONE is set there.
ARG TARGETARCH
ARG XAD_COMMIT=aaf4e9cad59cf1bfe9728b491f95f011db687f18
RUN wget https://github.com/auto-differentiation/xad/archive/${XAD_COMMIT}.tar.gz
RUN tar xf ${XAD_COMMIT}.tar.gz
RUN if [ "$TARGETARCH" = arm64 ]; then simd=NONE; else simd=AVX; fi \
 && cmake -S xad-${XAD_COMMIT} -B xad-${XAD_COMMIT}/build -DCMAKE_BUILD_TYPE=Release \
      -DXAD_ENABLE_TESTS=OFF -DXAD_WARNINGS_PARANOID=OFF -DXAD_SIMD_OPTION=$simd \
      -DXAD_POSITION_INDEPENDENT_CODE=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
      -DXAD_USE_STRONG_INLINE=ON -DXAD_NO_THREADLOCAL=ON -DXAD_HUGE_PAGES=ON \
      -DXAD_COPY_ELISION=ON -DXAD_SKIP_ZERO_PARTIALS=ON -DXAD_REDUCED_MEMORY=ON \
 && cmake --build xad-${XAD_COMMIT}/build --target install
ENV LIBRARY_PATH=/usr/local/lib

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/xad tools/xad

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "xad"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
