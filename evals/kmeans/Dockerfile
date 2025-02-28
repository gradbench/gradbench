FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget

RUN pip install numpy==1.26.0 pydantic dataclasses-json

WORKDIR /home/gradbench

COPY python /home/gradbench/python

COPY cpp /home/gradbench/cpp
RUN make -C cpp
COPY tools/manual/ /home/gradbench/tools/manual
RUN make -C /home/gradbench/tools/manual -Bj

ENV PYTHONPATH=/home/gradbench/python/gradbench:/home/gradbench/tools/
ENTRYPOINT ["python", "/home/gradbench/python/gradbench/gradbench/evals/kmeans/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
