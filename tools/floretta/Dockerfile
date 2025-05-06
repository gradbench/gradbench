FROM node
ARG TARGETARCH
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
COPY package.json .
COPY js/common js/common
COPY js/floretta js/floretta

# Create `node_modules`.
RUN npm install --omit dev

ENTRYPOINT ["node", "--disable-warning=ExperimentalWarning", "js/floretta/run.ts"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
