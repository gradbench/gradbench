FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget \
    libtool \
    automake \
    autoconf

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

# Install ADOL-C
RUN wget https://github.com/coin-or/ADOL-C/archive/refs/tags/releases/2.7.2.tar.gz
RUN tar xvf 2.7.2.tar.gz
RUN cd ADOL-C-releases-2.7.2 && autoreconf -fi
RUN cd ADOL-C-releases-2.7.2 && ./configure --enable-sparse --prefix=/usr/local
RUN cd ADOL-C-releases-2.7.2 && make install
ENV LIBRARY_PATH=/usr/local/lib64
ENV LD_LIBRARY_PATH=/usr/local/lib64

COPY cpp /gradbench/cpp
RUN make -C cpp

COPY tools/adol-c/ /gradbench/tools/adol-c

ENTRYPOINT ["python3", "/gradbench/tools/adol-c/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
