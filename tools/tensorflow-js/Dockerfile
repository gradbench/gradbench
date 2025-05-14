FROM oven/bun
WORKDIR /gradbench

# `COPY` the minimal set of files to create `node_modules`.
COPY package.json ./
COPY bun.lock ./
COPY js/common/package.json js/common/
COPY js/tensorflow/package.json js/tensorflow/
RUN bun install --omit dev

FROM node
COPY --from=0 /gradbench /gradbench
WORKDIR /gradbench

# `COPY` the rest of the files.
COPY js/common js/common
COPY js/tensorflow js/tensorflow

ENTRYPOINT ["node", "--disable-warning=ExperimentalWarning", "js/tensorflow/run.ts"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
