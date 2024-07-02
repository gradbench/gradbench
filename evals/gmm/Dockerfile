FROM python:alpine
WORKDIR /home/gradbench

COPY evals/gmm/ .

ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
