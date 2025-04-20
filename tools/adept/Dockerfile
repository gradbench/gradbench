FROM python:3.11-slim
WORKDIR /gradbench

# Install build dependencies.
RUN apt-get update && apt-get install -y automake build-essential wget

# Download and compile Adept.
#
# We copy in new versions of config.guess and config.sub because the
# versions in the tarball are obsolete and do not recognise
# new-fangled architectures such as ARM.
RUN wget http://www.met.reading.ac.uk/clouds/adept/adept-2.1.1.tar.gz
RUN tar xvf adept-2.1.1.tar.gz
RUN cp /usr/share/misc/config.guess /usr/share/misc/config.sub adept-2.1.1
RUN cd adept-2.1.1 && ./configure && make && make install
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
