FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py
RUN pip install tensorflow
RUN pip install scipy

WORKDIR /home/gradbench
COPY functions.py .
COPY run.py .

# shared
COPY shared/ .

# specific
COPY GMM/ .


ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
