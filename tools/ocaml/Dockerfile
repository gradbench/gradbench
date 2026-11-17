FROM debian:12-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    opam \
    libopenblas-dev \
    liblapacke-dev \
    zlib1g-dev \
    pkg-config

# Initialize opam, install OCaml 5.0.0.
RUN opam init --yes --disable-sandboxing && \
    opam update && \
    opam switch create 5.0.0

# Install dune.
RUN eval $(opam env) && opam install --yes dune=3.19.1

# Install yojson.
RUN eval $(opam env) && opam install --yes yojson

# Install owl.
#
# By default Owl compiles its C stubs with -march=native, which may result in
# instructions (like AVX-512) that are not available on the machine on which we
# run the Docker image. To ensure the binaries are reasonably portable, we pass
# Owl's default flags minus -march=native.
ARG TARGETARCH
RUN eval $(opam env) && \
    if [ "$TARGETARCH" = amd64 ]; then ARCH_CFLAGS="-mfpmath=sse -msse2"; fi && \
    OWL_CFLAGS="-g -O3 $ARCH_CFLAGS -funroll-loops -fno-math-errno -fno-rounding-math -fno-signaling-nans -fexcess-precision=fast -DSFMT_MEXP=19937 -fno-strict-aliasing" \
    opam install --yes owl=1.2

COPY tools/ocaml/ /gradbench/tools/ocaml

# Build tool.
RUN eval $(opam env) && dune build --root tools/ocaml --profile release

ENTRYPOINT ["/gradbench/tools/ocaml/_build/install/default/bin/gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
