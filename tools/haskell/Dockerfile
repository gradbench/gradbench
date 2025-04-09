FROM haskell:9

WORKDIR /gradbench

COPY tools/haskell/ /gradbench/tools/haskell

RUN cabal update

RUN cabal build --project-dir tools/haskell gradbench

ENTRYPOINT ["cabal", "run", "--project-dir", "tools/haskell", "gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
