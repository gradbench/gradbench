FROM oven/bun:1.2
WORKDIR /gradbench

# `COPY` the minimal set of files to create `node_modules`.
COPY package.json ./
COPY bun.lock ./
COPY js/common/package.json js/common/
COPY js/floretta/package.json js/floretta/
RUN bun install --omit dev

FROM node
ARG TARGETARCH
COPY --from=0 /gradbench /gradbench
WORKDIR /gradbench

# Prepare to install tools.
RUN mkdir bin
ENV PATH="/gradbench/bin:$PATH"
COPY tools/floretta/deps.js tools/floretta/

# Install `wasm-tools`.
RUN wget $(node tools/floretta/deps.js wasm-tools $TARGETARCH)
RUN tar xzf wasm-tools-* -C bin --strip-components=1

# Install Floretta.
RUN wget $(node tools/floretta/deps.js floretta $TARGETARCH)
RUN mv floretta-* bin/floretta
RUN chmod +x bin/floretta

# `COPY` the rest of the files.
COPY tools/floretta tools/floretta
COPY js/common js/common
COPY js/floretta js/floretta

ENTRYPOINT ["node", "--disable-warning=ExperimentalWarning", "js/floretta/run.ts"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
