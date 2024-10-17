FROM python:3.12-slim
WORKDIR /home/gradbench

COPY evals/gmm/ .

RUN pip install numpy==1.26.0

ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
