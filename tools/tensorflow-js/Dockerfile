FROM node
WORKDIR /gradbench

# `COPY` necessary files.
COPY package.json .
COPY js/common js/common
COPY js/tensorflow js/tensorflow

# Create `node_modules`.
RUN npm install --omit dev

ENTRYPOINT ["node", "--disable-warning=ExperimentalWarning", "js/tensorflow/run.ts"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
