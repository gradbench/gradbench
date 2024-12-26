FROM ubuntu
RUN apt-get update && apt-get install -y curl git jq
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
ENV PATH="/root/.elan/bin:$PATH"
WORKDIR /home/gradbench

COPY tools/scilean/lakefile.lean .
COPY tools/scilean/lake-manifest.json .
COPY tools/scilean/lean-toolchain .

RUN lake exe cache get      # Fetch the cache for dependencies

COPY tools/scilean/BuildSciLean.lean .
RUN lake build buildscilean # Build SciLean

COPY tools/scilean/Main.lean .
COPY tools/scilean/Gradbench.lean .
COPY tools/scilean/Gradbench/ ./Gradbench/
RUN lake build              # Build the SciLean gradbench tool

ENTRYPOINT ["/home/gradbench/.lake/build/bin/gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench