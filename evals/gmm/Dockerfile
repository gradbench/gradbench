FROM python:3.11-slim

RUN pip install numpy==1.26.0 pydantic scipy && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python/gradbench

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/gradbench/evals/gmm/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
