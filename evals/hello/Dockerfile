FROM python:alpine
WORKDIR /home/gradbench
COPY evals/hello/hello.adroit .
COPY evals/hello/main.py .
ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
