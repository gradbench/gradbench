FROM python:3.11-slim

WORKDIR /gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

COPY cpp /gradbench/cpp

RUN make -C cpp

COPY tools/manual/ /gradbench/tools/manual

ENTRYPOINT ["python3", "/gradbench/tools/manual/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
