FROM python:3.11-slim

RUN pip install numpy==1.26.0 dataclasses-json && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install scipy

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/pytorch/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
