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

ENTRYPOINT ["node", "tools/floretta/run.js"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
