FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py
RUN pip install tensorflow

WORKDIR /home/gradbench

COPY tools/tensorflow/ .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
