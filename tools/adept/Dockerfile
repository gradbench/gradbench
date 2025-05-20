FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y automake libtool build-essential wget

# Download and compile Adept.
RUN wget https://github.com/rjhogan/Adept-2/archive/refs/tags/v2.1.3.tar.gz
RUN tar xvf v2.1.3.tar.gz
RUN cd Adept-2-2.1.3 && autoreconf -i && ./configure && make && make install
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

# Download C++ dependencies.
COPY cpp cpp
RUN make -C cpp

# `COPY` the rest of the files.
COPY python python
COPY tools/adept tools/adept

ENTRYPOINT ["python3", "python/gradbench/gradbench/cpp.py", "adept"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
