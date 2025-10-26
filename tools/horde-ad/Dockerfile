FROM haskell:9

WORKDIR /gradbench

COPY tools/horde-ad/ /gradbench/tools/horde-ad

RUN cabal update

RUN cabal build --project-dir tools/horde-ad gradbench

ENTRYPOINT ["cabal", "run", "--project-dir", "tools/horde-ad", "gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
