FROM python:3.11-slim

RUN pip install jax jaxlib

WORKDIR /gradbench

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

ENTRYPOINT ["taskset", "-c", "1", "python", "/gradbench/python/gradbench/gradbench/tools/jax/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
