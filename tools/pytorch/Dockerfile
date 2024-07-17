FROM python:3.11-slim

RUN pip install --upgrade pip && \
    pip install numpy==1.26.0 && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install scipy

WORKDIR /home/gradbench

COPY tools/pytorch/ .
COPY shared/ADBench/ .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
