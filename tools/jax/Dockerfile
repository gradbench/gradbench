FROM python:3.11-slim

RUN pip install jax jaxlib

WORKDIR /home/gradbench

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python/gradbench

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/gradbench/jax/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
