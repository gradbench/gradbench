FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget \
    automake

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

# Download and compile Adept.
#
# We copy in new versions of config.guess and config.sub because the
# versions in the tarball are obsolete and do not recognise
# new-fangled architectures such as ARM.
ADD http://www.met.reading.ac.uk/clouds/adept/adept-2.1.1.tar.gz /gradbench
RUN tar xvf adept-2.1.1.tar.gz
RUN cp /usr/share/misc/config.guess /usr/share/misc/config.sub adept-2.1.1
RUN cd adept-2.1.1 && ./configure && make && make install
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY cpp /gradbench/cpp
RUN make -C cpp

COPY tools/adept/ /gradbench/tools/adept

ENTRYPOINT ["python3", "/gradbench/tools/adept/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
