FROM python:3.11-slim

WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget \
    curl

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python/gradbench

# Download and compile Adept.
#
# We download new versions of config.guess and config.sub because the
# versions in the tarball are obsolete and do not recognise
# new-fangled architectures such as ARM.
RUN curl -O http://www.met.reading.ac.uk/clouds/adept/adept-2.1.1.tar.gz
RUN tar xvf adept-2.1.1.tar.gz
RUN cd adept-2.1.1 && curl -O https://cvs.savannah.gnu.org/viewvc/*checkout*/config/config/config.guess
RUN cd adept-2.1.1 && curl -O https://cvs.savannah.gnu.org/viewvc/*checkout*/config/config/config.sub
RUN cd adept-2.1.1 && ./configure
RUN make -C adept-2.1.1
RUN make -C adept-2.1.1 install
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY cpp /home/gradbench/cpp
RUN make -C cpp

COPY tools/adept/ /home/gradbench/tools/adept

ENTRYPOINT ["python3", "/home/gradbench/tools/adept/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
