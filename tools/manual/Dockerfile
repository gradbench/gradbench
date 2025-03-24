FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y build-essential wget

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/manual tools/manual

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "manual"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
