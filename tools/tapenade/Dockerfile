FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    python3 \
    wget \
    openjdk-17-jre-headless

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

# install Tapenade
RUN wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
RUN tar -xvzf tapenade_3.16.tar

# ensure Tapenade is accessible
RUN ln -s /gradbench/tapenade_3.16/bin/tapenade /usr/local/bin/tapenade

COPY cpp /gradbench/cpp
RUN make -C cpp

COPY tools/tapenade/ /gradbench/tools/tapenade

ENTRYPOINT ["python3", "/gradbench/tools/tapenade/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
