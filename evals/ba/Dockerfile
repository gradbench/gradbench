FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget

RUN pip install numpy==1.26.0 pydantic

WORKDIR /gradbench

COPY python /gradbench/python

COPY cpp /gradbench/cpp
RUN make -C cpp
COPY tools/manual/ /gradbench/tools/manual
RUN make -C /gradbench/tools/manual -Bj

COPY evals/ba/data /gradbench/evals/ba/data

ENV PYTHONPATH=/gradbench/python/gradbench:/gradbench/tools/
ENTRYPOINT ["python", "/gradbench/python/gradbench/gradbench/evals/ba/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
