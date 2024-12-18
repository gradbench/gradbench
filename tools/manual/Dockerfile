FROM python:3.11-slim

WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    wget

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

COPY cpp /home/gradbench/cpp

RUN make -C cpp

COPY tools/manual/ /home/gradbench/tools/manual

RUN make -j -B -C tools/manual

ENTRYPOINT ["python3", "/home/gradbench/tools/manual/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
