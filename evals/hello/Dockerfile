FROM python:3.11-alpine

RUN pip install pydantic

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/evals/hello/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
