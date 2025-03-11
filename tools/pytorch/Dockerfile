FROM python:3.11-slim

RUN pip install numpy==1.26.0 dataclasses-json && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install scipy

COPY python /gradbench/python
ENV PYTHONPATH=/gradbench/python/gradbench

ENTRYPOINT ["python", "/gradbench/python/gradbench/gradbench/tools/pytorch/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
