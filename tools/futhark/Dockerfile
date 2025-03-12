FROM python:3.11-slim

RUN apt update
RUN apt install -y xz-utils build-essential git

RUN pip install futhark-data futhark-server dataclasses-json

ADD https://futhark-lang.org/releases/futhark-0.25.28-linux-x86_64.tar.xz futhark-0.25.28-linux-x86_64.tar.xz

RUN tar xvf futhark-0.25.28-linux-x86_64.tar.xz

RUN make -C futhark-0.25.28-linux-x86_64 install

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

COPY tools/futhark /gradbench/tools/futhark
RUN cd /gradbench/tools/futhark && futhark pkg sync

WORKDIR /gradbench
ENTRYPOINT ["python", "/gradbench/tools/futhark/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
