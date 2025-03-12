FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

# Install CoDiPack
RUN wget https://github.com/SciCompKL/CoDiPack/archive/refs/tags/v2.3.1.tar.gz
RUN tar xvf v2.3.1.tar.gz
ENV CPATH=/gradbench/CoDiPack-2.3.1/include

COPY cpp /gradbench/cpp
RUN make -C cpp

COPY tools/codipack/ /gradbench/tools/codipack

ENTRYPOINT ["python3", "/gradbench/tools/codipack/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
