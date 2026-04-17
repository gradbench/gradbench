FROM haskell:9.14

WORKDIR /gradbench

COPY tools/horde-ad/ /gradbench/tools/horde-ad

RUN cabal update

RUN cabal build --allow-newer --project-dir tools/horde-ad gradbench

ENTRYPOINT ["cabal", "run", "--allow-newer", "--project-dir", "tools/horde-ad", "gradbench"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
