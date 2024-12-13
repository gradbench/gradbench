FROM python:3.11-slim

WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget

# Download and compile Adept.
RUN wget http://www.met.reading.ac.uk/clouds/adept/adept-2.1.1.tar.gz
RUN tar xvf adept-2.1.1.tar.gz
RUN cd adept-2.1.1 && ./configure && make
ENV ADEPT_DIR=/home/gradbench/adept-2.1.1

COPY cpp /home/gradbench/cpp

RUN make -C cpp

COPY tools/adept/ /home/gradbench/tools/adept

RUN make -j -B -C tools/adept

ENTRYPOINT ["python3", "/home/gradbench/tools/adept/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
