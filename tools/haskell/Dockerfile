FROM haskell:9

WORKDIR /home/gradbench

COPY tools/haskell/ /home/gradbench/tools/haskell

RUN cabal update

RUN cabal build --project-dir tools/haskell gradbench

ENTRYPOINT ["cabal", "run", "--project-dir", "tools/haskell", "gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
