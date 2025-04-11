FROM node
WORKDIR /gradbench

COPY tools/floretta tools/floretta

ENTRYPOINT ["node", "tools/floretta/run.js"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
