FROM python:alpine
WORKDIR /home/gradbench

COPY evals/gmm/d2_k5.txt .
COPY evals/gmm/main.py .

ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
