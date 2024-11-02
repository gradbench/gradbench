FROM python:3.12-slim

RUN pip install numpy==1.26.0 pydantic

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

RUN ln -s /home/gradbench/python/gradbench/evals/gmm/validate.py /home/gradbench/validate

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/evals/gmm/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
