FROM python:3.11-slim

RUN pip install numpy==1.26.0 && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install scipy

COPY python /home/gradbench/python
ENV PYTHONPATH=/home/gradbench/python

ADD --chmod=0755 https://github.com/adroit-lang/adroit/releases/download/v0.2.2/adroit-x86_64-unknown-linux-musl /usr/local/bin/adroit

ENTRYPOINT ["python", "/home/gradbench/python/gradbench/pytorch/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
