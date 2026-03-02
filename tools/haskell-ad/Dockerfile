FROM haskell:9

WORKDIR /gradbench

COPY tools/haskell-ad/ /gradbench/tools/haskell-ad

RUN cabal update

RUN cabal build --project-dir tools/haskell-ad gradbench

ENTRYPOINT ["cabal", "run", "--project-dir", "tools/haskell-ad", "gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
