FROM python:3.11
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev
RUN pip install h5py
RUN pip install tensorflow-aarch64
WORKDIR /home/gradbench
COPY tensor_functions.py .
COPY tensor_run.py .
ENTRYPOINT ["python", "/home/gradbench/tensor_run.py"]
