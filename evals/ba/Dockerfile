FROM python:slim
WORKDIR /home/gradbench

COPY evals/ba/ .

ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
