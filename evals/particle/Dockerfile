FROM python:3.11-slim

RUN pip install numpy==1.26.0 pydantic dataclasses-json

WORKDIR /gradbench

COPY python /gradbench/python

ENV PYTHONPATH=/gradbench/python/gradbench:/gradbench/tools/
ENTRYPOINT ["python", "/gradbench/python/gradbench/gradbench/evals/particle/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
