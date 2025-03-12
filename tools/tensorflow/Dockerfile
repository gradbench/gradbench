FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py tensorflow scipy dataclasses-json

WORKDIR /gradbench

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

ENTRYPOINT ["python", "/gradbench/python/gradbench/gradbench/tools/tensorflow/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
