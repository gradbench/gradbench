FROM ghcr.io/gradbench/mathlib4:93904a0d036e5321062ae5e705e8a893da24d81b
COPY tools/scilean/lakefile.lean .
COPY tools/scilean/lake-manifest.json .
COPY tools/scilean/Main.lean .
RUN lake build
ENTRYPOINT ["/home/gradbench/.lake/build/bin/gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
