FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py tensorflow scipy

WORKDIR /home/gradbench

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/tensorflow/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
