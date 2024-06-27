FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py
RUN pip install tensorflow
RUN pip install scipy

WORKDIR /home/gradbench
COPY tensor_functions.py .
COPY tensor_run.py .

# shared
COPY shared/ .

# specific
COPY GMM/ .


ENTRYPOINT ["python", "/home/gradbench/tensor_run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
