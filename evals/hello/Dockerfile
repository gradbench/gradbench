FROM python:3.11-slim

RUN pip install numpy==1.26.0 pydantic

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

ENTRYPOINT ["python", "/gradbench/python/gradbench/gradbench/evals/hello/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
