FROM python:3.11-slim
RUN pip install numpy==1.26.0
RUN pip install mygrad
WORKDIR /home/gradbench

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/mygrad/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
