FROM python:3.11-slim
WORKDIR /home/gradbench

RUN pip install numpy==1.26.0
RUN pip install autograd

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/autograd/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
