FROM python:slim
WORKDIR /home/gradbench

RUN pip install numpy==1.26.0
COPY evals/ba/ .

ENTRYPOINT ["python3", "/home/gradbench/main.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
