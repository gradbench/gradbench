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
RUN eval $(opam env) && opam install --yes dune=3.7.0

# Install dune.
RUN eval $(opam env) && opam install --yes yojson

# Install owl.
RUN eval $(opam env) && opam install --yes owl=1.2

COPY tools/ocaml/ /gradbench/tools/ocaml

# Build tool.
RUN eval $(opam env) && dune build --root tools/ocaml --profile release

ENTRYPOINT ["/gradbench/tools/ocaml/_build/install/default/bin/gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
