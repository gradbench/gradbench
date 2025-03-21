FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y bash build-essential openjdk-17-jre-headless wget

# Install Tapenade.
RUN wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
RUN tar -xvzf tapenade_3.16.tar

# Ensure Tapenade is accessible.
RUN ln -s tapenade_3.16/bin/tapenade /usr/local/bin/tapenade

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/tapenade tools/tapenade

ENTRYPOINT ["python", "python/gradbench/gradbench/cpp.py", "tapenade"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
