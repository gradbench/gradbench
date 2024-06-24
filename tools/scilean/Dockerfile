FROM ghcr.io/gradbench/mathlib4:93904a0d036e5321062ae5e705e8a893da24d81b
COPY lakefile.lean .
COPY lake-manifest.json .
COPY Main.lean .
RUN lake build
ENTRYPOINT ["/home/gradbench/.lake/build/bin/gradbench"]
