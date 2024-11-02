FROM python:slim

RUN pip install pydantic

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

RUN ln -s /home/gradbench/python/gradbench/evals/ba/validate.py /home/gradbench/validate

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/evals/ba/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
